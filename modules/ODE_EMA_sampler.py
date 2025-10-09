# auto_cone_dir_ode_opt.py
from __future__ import annotations
import torch
import torch.nn.functional as F
import comfy.samplers

# -------------------- cached spatial kernels (module scope) --------------------
_SOBEL_KX_BASE = torch.tensor([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_KY_BASE = torch.tensor([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
_LAPLACE_BASE  = torch.tensor([[0, 1, 0],
                               [1,-4, 1],
                               [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

# ---------- small helpers (no per-step kernel allocations) ----------
def _avg_blur(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    if x.dim() != 4 or k <= 1:
        return x
    pad = k // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    return F.avg_pool2d(x, kernel_size=k, stride=1)

def _sobel_mag(x: torch.Tensor) -> torch.Tensor:
    """
    Depthwise Sobel gradient magnitude.
    Kernels are cached; compute in fp32 for stability, cast back to input dtype.
    """
    if x.dim() != 4:
        return torch.zeros_like(x)
    B, C, H, W = x.shape
    dev = x.device

    x32 = x.float()
    kx = _SOBEL_KX_BASE.to(dev).expand(C, 1, 3, 3).contiguous()
    ky = _SOBEL_KY_BASE.to(dev).expand(C, 1, 3, 3).contiguous()

    gx = F.conv2d(x32, kx, padding=1, groups=C)
    gy = F.conv2d(x32, ky, padding=1, groups=C)
    eps = torch.finfo(gx.dtype).eps
    g32 = torch.sqrt(gx.square() + gy.square() + eps)
    return g32.to(x.dtype)

def _laplace(x: torch.Tensor) -> torch.Tensor:
    """Depthwise 3x3 Laplacian in fp32, cast back to x dtype."""
    if x.dim() != 4:
        return torch.zeros_like(x)
    C = x.shape[1]
    dev = x.device
    x32 = x.float()
    k = _LAPLACE_BASE.to(dev).expand(C, 1, 3, 3).contiguous()
    return F.conv2d(x32, k, padding=1, groups=C).to(x.dtype)

def _unit_var(z: torch.Tensor) -> torch.Tensor:
    v = z.pow(2).mean(dim=(1, 2, 3), keepdim=True).clamp_min(torch.finfo(z.dtype).eps)
    return z * torch.rsqrt(v)

# ---------- core: ODE in σ-space, with detail options ----------
def _make_auto_cone_dir_ode_opt(
    stats_downsample: int = 64,
    # Detail emphasis (edge-aware gain)
    detail_gain_max: float = 0.35,          # 0 disables; typical 0.15–0.35
    detail_power: float = 0.5,              # response to edges; 0.8–1.5 reasonable
    detail_schedule_power: float = 4.0,     # late-step emphasis; 1–3
):
    """
    model(x, sigma) -> x0 (denoised) per k-diff convention.
    Update (ODE): x_{i+1} = x_i + (σ_{i+1}-σ_i) * d̂_i,  d_i = (x_i - x0)/σ_i.
    We shape d_i with a smooth SPD cone A (EMA-stabilized), then reweight for detail,
    renormalize, and optionally apply a bounded shock sharpen near the end.

    Detail controls:
      - detail_gain_max: max multiplicative gain on high-frequency directions (0 disables).
      - detail_power: nonlinearity on the normalized gradient map.
      - detail_schedule_power: how strongly the gain ramps up as σ→0.
      - shock_strength / shock_last_steps: late, bounded shock-like sharpening.
      - heun_corrector_last_steps: late second-order corrector using the same A.
    """

    # sanitize once for closure
    stats_downsample     = max(1, int(stats_downsample))
    detail_gain_max      = max(0.0, float(detail_gain_max))
    detail_power         = max(0.1, float(detail_power))
    detail_schedule_power= max(0.1, float(detail_schedule_power))


    @torch.no_grad()
    def _sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise=None, **kwargs):
        extra_args = {} if extra_args is None else extra_args
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # Move entire schedule to x's device/dtype once; ensure descending schedule
        sig = sigmas.to(device=device, dtype=dtype)
        sig = sig.flip(0) if sig[0] < sig[-1] else sig
        steps = len(sig) - 1
        sigma0 = sig[0]
        sigma = sigma0.expand(B)

        # EMA for cone to kill crawling texture
        A_prev = None
        CONE_EMA = 0.7
        A_MIN    = 0.75   # keep anisotropy gentle, SPD & bounded
        BETA_MAX = 0.45   # max cone strength (small; avoids streaking)

        # clamp effective downsample to spatial size
        sd_eff = max(1, min(stats_downsample, int(H), int(W)))

        for i in range(steps):
            sigma_next = sig[i + 1].expand(B)

            # 1) predict denoised x0 (do NOT clamp/filter it)
            x0 = model(x, sigma, **extra_args)

            # 2) base direction in σ-space
            d = (x - x0) / sigma.view(B, 1, 1, 1)

            # 3) build cone A(x0) with shared gradient stats
            g  = _avg_blur(_sobel_mag(x0), k=3)      # (B,C,H,W)

            # downsample for robust stats (major cost win on high res)
            if sd_eff > 1:
                g_stats = F.avg_pool2d(g, kernel_size=sd_eff, stride=sd_eff)
            else:
                g_stats = g

            # robust median & MAD in fp32 for stability
            flat = g_stats.view(B, -1).float()
            med_scalar = flat.median(dim=1).values                                   # (B,)
            mad_scalar = (flat - med_scalar.unsqueeze(1)).abs().median(dim=1).values # (B,)

            med = med_scalar.view(B, 1, 1, 1).to(dtype)
            scale = (mad_scalar.view(B, 1, 1, 1) * 1.4826).clamp_min(1e-6).to(dtype)

            # normalised gradient magnitude (non-negative)
            gn_base = (g - med) / scale
            gn = gn_base.clamp_min(0.0)

            # shared robust HF proxy from same stats
            s = gn_base.abs().mean(dim=(1, 2, 3), keepdim=True)
            hf = (s / (1.0 + s)).detach()

            sigma_frac = (sigma / sigma0).clamp(0, 1).view(B, 1, 1, 1)
            beta = (BETA_MAX * (0.5 * sigma_frac + 0.5 * hf)).to(dtype)

            # SPD cone, bounded; rsqrt for fewer temps
            A_now = torch.rsqrt(1.0 + beta * gn.square()).clamp(min=A_MIN, max=1.0)

            # EMA smoothing over steps
            A_smooth = A_now if A_prev is None else (CONE_EMA * A_now + (1.0 - CONE_EMA) * A_prev)
            A_prev = A_smooth

            # 4) detail emphasis (edge-aware gain), ramps up as σ→0
            if detail_gain_max > 0.0:
                progress = (1.0 - sigma_frac)  # 0 at start, 1 near end
                gamma = (detail_gain_max * (progress ** detail_schedule_power)).to(dtype)
                W_detail = 1.0 + gamma * (gn.clamp_min(0.0) ** detail_power)
            else:
                W_detail = 1.0

            d_shaped = A_smooth * d
            d_mod = _unit_var(W_detail * d_shaped)

            # Step size
            h = (sigma_next - sigma).view(B, 1, 1, 1)
            x_next = x + h * d_mod
            x, sigma = x_next, sigma_next

            if callback is not None:
                callback({
                    'x': x, 'i': i,
                    'sigma': sigma, 'sigma_hat': sigma,
                    'denoised': x0,
                })

        return x

    return _sampler

# ---------- Comfy nodes (UI-exposed knobs) ----------
class ODE_EMA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Existing controls
                "stats_downsample": ("INT", {"default": 64, "min": 1, "max": 64, "step": 1}),
                # Detail emphasis
                "detail_gain_max": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.5, "step": 0.01}),
                "detail_power": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.5, "step": 0.01}),
                "detail_schedule_power": ("FLOAT", {"default": 4.0, "min": 2.0, "max": 4.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = "MilitantAI/Switchblade/Generation"

    def build(self,
              stats_downsample: int,
              detail_gain_max: float,
              detail_power: float,
              detail_schedule_power: float):
        sampler_fn = _make_auto_cone_dir_ode_opt(
            stats_downsample=stats_downsample,
            detail_gain_max=detail_gain_max,
            detail_power=detail_power,
            detail_schedule_power=detail_schedule_power,
        )
        return (comfy.samplers.KSAMPLER(sampler_fn),)

NODE_CLASS_MAPPINGS = {
    "ODE-EMA Sampler": ODE_EMA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ODE-EMA Sampler": "ODE EMA Sampler",
}
