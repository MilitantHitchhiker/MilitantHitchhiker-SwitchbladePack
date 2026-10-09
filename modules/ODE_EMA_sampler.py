from __future__ import annotations

import math
import torch
import torch.nn.functional as F
import comfy.samplers
from comfy.k_diffusion import sampling as k_diffusion_sampling


# -----------------------------------------------------------------------------
# Local anisotropic history for rectified-flow / Anima-style latents
# -----------------------------------------------------------------------------

_FLOW_ETA = 1e-3
_FLOW_EPS_V = 1e-4


def _as_5d(x: torch.Tensor):
    """Return [B,C,T,H,W] plus whether a synthetic T dimension was added."""
    if x.dim() == 5:
        return x, False
    if x.dim() == 4:
        return x.unsqueeze(2), True
    raise ValueError(
        f"ODE EMA Flow expects a 4D or 5D latent, got shape {tuple(x.shape)}"
    )


def _pad_hw(x5: torch.Tensor, pool: int):
    """Replicate-pad H/W to a multiple of pool."""
    H, W = x5.shape[-2:]
    Hp = int(math.ceil(H / pool) * pool)
    Wp = int(math.ceil(W / pool) * pool)
    ph = Hp - H
    pw = Wp - W
    if ph or pw:
        # x5 is always 5D [B,C,T,H,W] here. For non-constant padding,
        # PyTorch's 5D path expects padding for the final three dimensions:
        # (W_left, W_right, H_top, H_bottom, T_front, T_back).
        # We only pad H/W and leave T unchanged.
        x5 = F.pad(x5, (0, pw, 0, ph, 0, 0), mode="replicate")
    return x5, H, W


def _to_blocks(x5: torch.Tensor, pool: int):
    """
    [B,C,T,H,W] -> [B,Hb,Wb,C,N], N=T*pool*pool.
    """
    B, C, T, H, W = x5.shape
    Hb = H // pool
    Wb = W // pool
    blocks = (
        x5.reshape(B, C, T, Hb, pool, Wb, pool)
          .permute(0, 3, 5, 1, 2, 4, 6)
          .reshape(B, Hb, Wb, C, T * pool * pool)
    )
    return blocks


def _from_blocks(blocks: torch.Tensor, T: int, pool: int, H: int, W: int):
    """
    [B,Hb,Wb,C,N] -> [B,C,T,H,W], cropping any pad.
    """
    B, Hb, Wb, C, N = blocks.shape
    x5 = (
        blocks.reshape(B, Hb, Wb, C, T, pool, pool)
              .permute(0, 3, 4, 1, 5, 2, 6)
              .reshape(B, C, T, Hb * pool, Wb * pool)
    )
    return x5[..., :H, :W]


def _local_channel_operator(
    x: torch.Tensor,
    x0: torch.Tensor,
    pool: int,
    eta: float = _FLOW_ETA,
):
    """
    Build a local pooled channel second-moment operator.

        d = x - x0
        S_j = (1/N) sum_{p in block j} d_p d_p^T
        A_j = S_j / (tr(S_j) + eta^2)

    Each spatial block gets its own [C,C] operator.  T is pooled together with
    the H/W neighborhood.  The result is [B,Hb,Wb,C,C].
    """
    d5, _ = _as_5d((x - x0).float())
    d5, H, W = _pad_hw(d5, pool)
    db = _to_blocks(d5, pool)

    N = max(int(db.shape[-1]), 1)
    S = torch.einsum("bhwcn,bhwdn->bhwcd", db, db) / float(N)
    tr = torch.diagonal(S, dim1=-2, dim2=-1).sum(dim=-1)[..., None, None]

    A = S / (tr + float(eta) ** 2).clamp_min(torch.finfo(S.dtype).eps)
    return A, H, W


def _update_history(
    A_inst: torch.Tensor,
    A_prev: torch.Tensor | None,
    sigma: torch.Tensor,
    sigma_prev: torch.Tensor | None,
    history_length: float,
):
    """
    Exponential relaxation in noise-level distance:

        h = |sigma_n - sigma_{n-1}|
        beta = exp(-h / ell)
        A_bar = beta A_prev + (1-beta) A_inst
    """
    if A_prev is None or sigma_prev is None:
        return A_inst

    ell = max(float(history_length), 1e-8)
    h = (sigma.float() - sigma_prev.float()).abs()
    beta = torch.exp(-h / ell).view(-1, 1, 1, 1, 1)
    return beta * A_prev + (1.0 - beta) * A_inst


def _orthogonal_local_turn(
    v: torch.Tensor,
    A_bar: torch.Tensor,
    turn_strength: float,
    pool: int,
    eps_v: float = _FLOW_EPS_V,
):
    """
    Exact local orthogonal correction:

        K = 2 A_bar - I
        q = (||v||^2 K v - v(v^T K v)) / (||v||^2 + eps_v^2)
        v_tilde = v + alpha q

    alpha is exposed directly as turn_strength in [-1, 1].
    """
    v5, squeezed_t = _as_5d(v.float())
    v5p, H, W = _pad_hw(v5, pool)
    B, C, T, Hp, Wp = v5p.shape

    vb = _to_blocks(v5p, pool)

    Av = torch.einsum("bhwcd,bhwdn->bhwcn", A_bar.float(), vb)
    Kv = 2.0 * Av - vb

    r = (vb * vb).sum(dim=3, keepdim=True)
    vKv = (vb * Kv).sum(dim=3, keepdim=True)

    q = (r * Kv - vb * vKv) / (r + float(eps_v) ** 2)

    alpha = max(-1.0, min(1.0, float(turn_strength)))
    proposal = vb + alpha * q

    out5 = _from_blocks(proposal, T=T, pool=pool, H=H, W=W)
    out = out5.squeeze(2) if squeezed_t else out5

    tiny = torch.finfo(vb.dtype).eps
    vnorm = torch.sqrt(r.clamp_min(tiny))
    qnorm = torch.sqrt((q * q).sum(dim=3, keepdim=True).clamp_min(0.0))
    rel = qnorm / vnorm

    return out.to(dtype=v.dtype), float(rel.mean().item()), float(rel.max().item())


def _make_flow_sampler(
    turn_strength: float = 0.35,
    history_length: float = 0.15,
    pool_size: int = 8,
):
    turn_strength = max(-1.0, min(1.0, float(turn_strength)))
    history_length = max(1e-8, float(history_length))
    pool_size = max(2, int(pool_size))

    @torch.no_grad()
    def _sampler(
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        noise=None,
        **kwargs,
    ):
        extra_args = {} if extra_args is None else extra_args

        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        sig = sigmas.to(device=device, dtype=dtype)
        sig = sig.flip(0) if sig[0] < sig[-1] else sig
        steps = len(sig) - 1

        A_bar = None
        sigma_prev = None

        for i in range(steps):
            sigma_scalar = sig[i]
            sigma_next_scalar = sig[i + 1]

            sigma = sigma_scalar.expand(B)
            sigma_next = sigma_next_scalar.expand(B)

            x0 = model(x, sigma, **extra_args)

            sigma_view = sigma.view((B,) + (1,) * (x.dim() - 1))
            tiny = torch.finfo(dtype).tiny
            v = (x - x0) / sigma_view.clamp_min(tiny)

            A_inst, _, _ = _local_channel_operator(
                x=x,
                x0=x0,
                pool=pool_size,
                eta=_FLOW_ETA,
            )
            A_bar = _update_history(
                A_inst=A_inst,
                A_prev=A_bar,
                sigma=sigma,
                sigma_prev=sigma_prev,
                history_length=history_length,
            )

            v_tilde, mean_rel, max_rel = _orthogonal_local_turn(
                v=v,
                A_bar=A_bar,
                turn_strength=turn_strength,
                pool=pool_size,
                eps_v=_FLOW_EPS_V,
            )

            if i == 0 or i == steps // 2 or i == steps - 1:
                print(
                    "[ODE EMA Flow] "
                    f"step={i}/{steps} alpha={turn_strength:.3f} "
                    f"pool={pool_size} q/v mean={mean_rel:.5f} max={max_rel:.5f}"
                )

            if callback is not None:
                callback({
                    "x": x,
                    "i": i,
                    "sigma": sigma_scalar,
                    "sigma_hat": sigma_scalar,
                    "denoised": x0,
                })

            h = (sigma_next - sigma).view((B,) + (1,) * (x.dim() - 1))
            x = x + h * v_tilde

            sigma_prev = sigma.detach()

        return x

    return _sampler


class _FlowSamplerWrapper(comfy.samplers.Sampler):
    """
    Rebuild the `simple` schedule before KSAMPLER performs flow noise scaling,
    then run the custom rectified-flow field.
    """
    def __init__(self, sampler_fn):
        self.inner = comfy.samplers.KSAMPLER(sampler_fn)

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        steps = max(int(len(sigmas) - 1), 1)
        model_sampling = model_wrap.inner_model.model_sampling
        flow_sigmas = comfy.samplers.calculate_sigmas(
            model_sampling, "simple", steps
        ).cpu()

        return self.inner.sample(
            model_wrap,
            flow_sigmas,
            extra_args,
            callback,
            noise,
            latent_image=latent_image,
            denoise_mask=denoise_mask,
            disable_pbar=disable_pbar,
        )



# -----------------------------------------------------------------------------
# Stock KSampler registration
# -----------------------------------------------------------------------------

ODE_EMA_SAMPLER_NAME = "ode_ema_flow"

ODE_EMA_FIXED_TURN_STRENGTH = 0.35
ODE_EMA_FIXED_HISTORY_LENGTH = 0.15
ODE_EMA_FIXED_POOL_SIZE = 8


@torch.no_grad()
def sample_ode_ema_flow(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    **kwargs,
):
    """
    Fixed-parameter stock-KSampler entry.

    Canonical settings:
        turn_strength = 0.35
        history_length = 0.15
        pool_size = 8

    For Anima / flow models, use scheduler="simple".
    """
    sampler_fn = _make_flow_sampler(
        turn_strength=ODE_EMA_FIXED_TURN_STRENGTH,
        history_length=ODE_EMA_FIXED_HISTORY_LENGTH,
        pool_size=ODE_EMA_FIXED_POOL_SIZE,
    )
    return sampler_fn(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        **kwargs,
    )


def _register_ode_ema_sampler():
    """
    Register ODE EMA Flow in ComfyUI's live sampler lists without editing core
    files. The existing tunable custom SAMPLER node remains available.
    """
    setattr(
        k_diffusion_sampling,
        f"sample_{ODE_EMA_SAMPLER_NAME}",
        sample_ode_ema_flow,
    )

    # Mutate live lists in place so existing references see the new entry.
    for attr in ("SAMPLER_NAMES", "KSAMPLER_NAMES"):
        names = getattr(comfy.samplers, attr, None)
        if isinstance(names, list) and ODE_EMA_SAMPLER_NAME not in names:
            names.append(ODE_EMA_SAMPLER_NAME)

    samplers = getattr(comfy.samplers.KSampler, "SAMPLERS", None)
    if isinstance(samplers, list) and ODE_EMA_SAMPLER_NAME not in samplers:
        samplers.append(ODE_EMA_SAMPLER_NAME)


_register_ode_ema_sampler()

class ODE_EMA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "turn_strength": (
                    "FLOAT",
                    {"default": 0.35, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "history_length": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.001, "max": 2.0, "step": 0.01},
                ),
                "pool_size": (
                    "INT",
                    {"default": 8, "min": 2, "max": 64, "step": 2},
                ),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = "MilitantAI/Switchblade/Generation"

    def build(
        self,
        turn_strength: float,
        history_length: float,
        pool_size: int,
    ):
        sampler_fn = _make_flow_sampler(
            turn_strength=turn_strength,
            history_length=history_length,
            pool_size=pool_size,
        )
        return (_FlowSamplerWrapper(sampler_fn),)


NODE_CLASS_MAPPINGS = {
    "ODE-EMA Sampler": ODE_EMA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ODE-EMA Sampler": "ODE EMA Flow Sampler",
}
