# ComfyUI custom nodes: GOD Sampler (SAMPLER)
# Drop this file into ComfyUI/custom_nodes/ and restart.

import torch
import torch.nn.functional as F
import comfy.samplers  # to return a KSAMPLER

# =========================
# Helpers: shapes & padding
# =========================

def _as_nchw(t):
    if t is None:
        return None
    if t.ndim == 4:
        return t
    if t.ndim == 3:         # [C,H,W] -> [1,C,H,W]
        return t.unsqueeze(0)
    if t.ndim == 2:         # [H,W]   -> [1,1,H,W]
        return t.unsqueeze(0).unsqueeze(0)
    if t.ndim == 1:         # [W]     -> [1,1,1,W]
        return t.view(1, 1, 1, -1)
    raise ValueError(f"Expected <=4D tensor for image ops, got {t.shape}")

def _pad2d(img, pad, prefer="reflect"):
    # pad=(left,right,top,bottom); switch to 'replicate' if H or W == 1
    h, w = img.shape[-2], img.shape[-1]
    mode = prefer if (h > 1 and w > 1) else "replicate"
    return F.pad(img, pad, mode=mode)

# =========================
# GOD Tensor helper math
# =========================

def _rho(x):
    return (x**2).mean(dim=1, keepdim=True)

def _grad_xy(img):
    img = _as_nchw(img)
    pad_h = _pad2d(img, (0, 0, 0, 1), prefer="reflect")  # +1 bottom (H)
    pad_w = _pad2d(img, (0, 1, 0, 0), prefer="reflect")  # +1 right  (W)
    gx = pad_h[..., 1:, :] - img     # forward diff along H
    gy = pad_w[..., :, 1:] - img     # forward diff along W
    return gx, gy

def _norm2(gx, gy, eps=1e-12):
    return torch.sqrt(gx*gx + gy*gy + eps)

def _unit(vx, vy, eps=1e-12):
    n = _norm2(vx, vy, eps)
    return vx/n, vy/n, n

def _argmax_2d(t):
    B, _, H, W = t.shape
    flat = t.reshape(B, -1)  # safe for non-contiguous tensors
    idx = flat.argmax(dim=1)
    y = (idx // W).clamp(max=H-1)
    x = (idx %  W)
    return torch.stack([y, x], dim=1)

def _gather_patch(arr, centers, r=1):
    B, C, H, W = arr.shape
    ys = centers[:,0].clamp(r, H-1-r)
    xs = centers[:,1].clamp(r, W-1-r)
    patches = []
    for b in range(B):
        y, x = int(ys[b]), int(xs[b])
        patches.append(arr[b:b+1, :, y-r:y+r+1, x-r:x+r+1])
    return torch.cat(patches, dim=0)

def _softmax_max(mag_patch, beta=50.0):
    B = mag_patch.shape[0]
    w = torch.softmax(beta * mag_patch.view(B, -1), dim=1)
    return (w * mag_patch.view(B, -1)).sum(dim=1, keepdim=True).view(B,1,1,1)

def _maxpool3x3(x):
    x = _as_nchw(x)
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

def _mean_masked(v, m, eps=1e-8):
    spatial = tuple(range(2, v.ndim))
    return (v*m).sum(dim=spatial, keepdim=True) / (m.sum(dim=spatial, keepdim=True) + eps)

def _ray_integral(rho, centers, alpha_level):
    B, _, H, W = rho.shape
    dirs = torch.tensor([[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]],
                        device=rho.device, dtype=torch.long)
    out = []
    for b in range(B):
        y0, x0 = int(centers[b,0]), int(centers[b,1])
        acc = 0.0
        for d in dirs:
            y, x = y0, x0
            last = rho[b,0,y,x].item()
            for _ in range(H+W):
                y = max(0, min(H-1, y + int(d[0])))
                x = max(0, min(W-1, x + int(d[1])))
                val = rho[b,0,y,x].item()
                acc += val
                a = alpha_level[b,0,0,0].item()
                if (last >= a and val <= a) or (last <= a and val >= a):
                    break
                last = val
        out.append(acc / 8.0)
    return torch.tensor(out, device=rho.device).view(B,1,1,1)

def _orientation_stress_and_unity(rho_cur, rho_prev):
    gx, gy = _grad_xy(rho_cur)
    ux, uy, _ = _unit(gx, gy)
    gxp, gyp = _grad_xy(rho_prev)
    uxp, uyp, _ = _unit(gxp, gyp)
    ups = torch.mean(torch.abs(ux*(ux-uxp) + uy*(uy-uyp)), dim=(2,3), keepdim=True) + 1e-12
    uni = torch.mean((ux*uxp + uy*uyp), dim=(2,3), keepdim=True)
    return ups, uni

# =========================
# Synergy (uses all helpers)
# =========================

def _synergy_state(x_cur, x_prev):
    x_cur = _as_nchw(x_cur)
    x_prev = _as_nchw(x_prev) if x_prev is not None else None

    rho = _rho(x_cur)                    # [B,1,H,W]
    gx, gy = _grad_xy(rho)
    ux, uy, gmag = _unit(gx, gy)

    # Canonical C_Z via peak neighborhood
    centers  = _argmax_2d(rho)                     # [B,2]
    mag_patch = _gather_patch(gmag, centers, r=1)  # [B,1,3,3]
    C_Z = _softmax_max(mag_patch, beta=50.0)       # [B,1,1,1]

    # Φ via interior mask around α-level
    rho_max = rho.amax(dim=(2,3), keepdim=True)
    alpha_level = 0.5 * rho_max
    M = torch.sigmoid((rho - alpha_level)/1e-3)
    Minv = 1.0 - M
    rho_in  = _mean_masked(rho, M)
    rho_out = _mean_masked(rho, Minv)
    delta = rho_in / (rho_in + rho_out + 1e-8)
    Phi = 2.0*delta - 1.0

    # Θ via ray integral from the peak to the α-level horizon
    Theta = torch.exp(-_ray_integral(rho, centers, alpha_level))

    # Υ, U from prior frame
    if x_prev is None:
        Upsilon = torch.ones_like(C_Z) * 1e-6
        U = torch.ones_like(C_Z)
    else:
        rho_p = _rho(x_prev)
        Upsilon, U = _orientation_stress_and_unity(rho, rho_p)

    # Creation count & entropic resistance
    Csoft  = torch.sigmoid((rho - _maxpool3x3(rho))/1e-3)
    Ccount = Csoft.sum(dim=(2,3), keepdim=True)
    N = Ccount / (U + 1.0e-12)

    # Final synergy
    S = (C_Z * Phi.clamp_min(1e-12) * Theta.clamp_min(1e-12)) / \
        (Upsilon.clamp_min(1e-12) * N.clamp_min(1e-12))
    return S  # [B,1,1,1]

# =========================
# GOD-Flow sampler factory (exposes settings)
# =========================

def _make_sampler(gain: float):
    """
    Returns a sampler function bound to requested behavior.
    mode: 'off' | 'min' | 'blend'
    carry: if True, next step uses adapted sigma; else uses scheduled sigma.
    """
    gain = float(gain)
    blend = 0.5

    @torch.no_grad()
    def _sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        extra_args = {} if extra_args is None else extra_args

        x = _as_nchw(x)
        dtype = x.dtype
        device = x.device
        B = x.shape[0]

        # ensure descending schedule if needed
        if sigmas[0] < sigmas[-1]:
            sigmas_f = torch.flip(sigmas, dims=[0])
        else:
            sigmas_f = sigmas

        sigma_min = float(getattr(model, "sigma_min", 0.01))

        x_prev = None
        steps = len(sigmas_f) - 1

        # current sigma state
        s_cur = sigmas_f[0].to(device).expand(B)

        for i in range(steps):
            # 1) model call at chosen sigma (Comfy expects [B]-shape sigma)
            denoised = model(x, s_cur, **extra_args)

            # 2) eps using s_cur
            alpha_i = 1.0 / torch.sqrt(1.0 + s_cur**2)              # [B]
            eps = (x - alpha_i.view(B,1,1,1) * denoised) / s_cur.view(B,1,1,1)

            # 3) next sigma
            S = _synergy_state(x, x_prev)                         # [B,1,1,1]
            s_next_sched = sigmas_f[i+1].to(device).expand(B)     # [B]
            s_next_adapt = torch.clamp(
                s_cur.view(B,1,1,1) / (1.0 + gain * S),
                min=sigma_min
            ).view(B)                                             # [B]
            s_next = (1.0 - blend) * s_next_sched + blend * s_next_adapt

            # 4) deterministic projection to s_next
            alpha_j = 1.0 / torch.sqrt(1.0 + s_next**2)
            x0_hat  = denoised
            x_new   = alpha_j.view(B,1,1,1) * x0_hat + s_next.view(B,1,1,1) * eps

            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': s_cur, 'sigma_hat': s_cur, 'denoised': x0_hat})

            x_prev = x.detach()
            x = x_new.to(dtype).detach()
            s_cur = s_next.detach()

        return x

    return _sampler

# =========================
# ComfyUI Node returning a SAMPLER (with inputs)
# =========================

class GOD_Sampler_Advanced_Ext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "god_strength":("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = "MilitantAI/Switchblade/Generation"

    def build(self, god_strength):
        sampler_fn = _make_sampler(
            gain=float(god_strength),
        )
        sampler = comfy.samplers.KSAMPLER(sampler_fn)
        return (sampler,)

NODE_CLASS_MAPPINGS = {
    "GOD Sampler (Advanced) Ext.": GOD_Sampler_Advanced_Ext,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GOD Sampler (Advanced) Ext.": "GOD Sampler (Advanced) Ext.",
}
