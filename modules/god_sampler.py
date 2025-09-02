# ComfyUI custom nodes: GOD Sampler (SAMPLER)
# Drop this file into ComfyUI/custom_nodes/ and restart.

import torch
import torch.nn.functional as F
import comfy.samplers  # required to return a KSAMPLER wrapper

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
    flat = t.reshape(B, -1)  # reshape safe for non-contiguous tensors
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

def _synergy_state(x_cur, x_prev):
    x_cur = _as_nchw(x_cur)
    x_prev = _as_nchw(x_prev) if x_prev is not None else None

    rho = _rho(x_cur)                    # [B,1,H,W]
    gx, gy = _grad_xy(rho)
    ux, uy, gmag = _unit(gx, gy)

    B = rho.shape[0]
    w_peak = torch.softmax(50.0 * rho.reshape(B, -1), dim=1).view_as(rho)
    C_Z = (w_peak * gmag).sum(dim=(2,3), keepdim=True)

    rho_max = rho.amax(dim=(2,3), keepdim=True)
    M = torch.sigmoid((rho - 0.5*rho_max)/1e-3)
    Minv = 1.0 - M
    rho_in  = _mean_masked(rho, M)
    rho_out = _mean_masked(rho, Minv)
    delta = rho_in / (rho_in + rho_out + 1e-8)
    Phi = 2.0*delta - 1.0
    Theta = torch.exp(-_mean_masked(rho, M))

    if x_prev is None:
        Upsilon = torch.ones_like(C_Z) * 1e-6
        U = torch.ones_like(C_Z)
    else:
        rho_p = _rho(x_prev)
        gxp, gyp = _grad_xy(rho_p)
        uxp, uyp, _ = _unit(gxp, gyp)
        spatial = (2,3)
        Upsilon = torch.mean(torch.abs(ux*(ux-uxp)+uy*(uy-uyp)), dim=spatial, keepdim=True) + 1e-12
        U = torch.mean((ux*uxp+uy*uyp), dim=spatial, keepdim=True)

    Csoft  = torch.sigmoid((rho - _maxpool3x3(rho))/1e-3)
    Ccount = Csoft.sum(dim=(2,3), keepdim=True)
    N = Ccount / (U + 1e-12)

    S = (C_Z * Phi.clamp_min(1e-12) * Theta.clamp_min(1e-12)) / \
        (Upsilon.clamp_min(1e-12) * N.clamp_min(1e-12))
    return S  # [B,1,1,1]

# =========================
# GOD-Flow core sampler (k-diffusion style)
# =========================

@torch.no_grad()
def god_flow_sample(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    """
    Signature matches ComfyUI's samplers (see nodes_advanced_samplers.py).
    - model(x, sigma_hat * s_in, **extra_args)
    - sigmas: 1D tensor of length steps
    """
    extra_args = {} if extra_args is None else extra_args

    x = _as_nchw(x)
    device = x.device
    dtype = x.dtype
    B = x.shape[0]
    s_in = x.new_ones([B])  # per Comfy convention for sigma vector broadcasting  :contentReference[oaicite:1]{index=1}

    # ensure descending schedule if needed
    if sigmas[0] < sigmas[-1]:
        sigmas = torch.flip(sigmas, dims=[0])

    sigma_min = float(getattr(model, "sigma_min", 0.01))

    x_prev = None
    steps = len(sigmas) - 1
    for i in range(steps):
        sigma_i = sigmas[i]
        sigma_next_sched = sigmas[i+1]

        # call model with [B] sigma vector
        denoised = model(x, sigma_i * s_in, **extra_args)  # Comfy expects this call form  :contentReference[oaicite:2]{index=2}

        # derive epsilon from denoised (standard k-diffusion relation)
        # x = alpha * x0 + sigma * eps  =>  eps = (x - alpha*x0) / sigma
        alpha_i = 1.0 / torch.sqrt(1.0 + (sigma_i*s_in)**2)
        eps = (x - alpha_i.view(B,1,1,1) * denoised) / (sigma_i*s_in).view(B,1,1,1)

        # synergy-adapted next sigma
        S = _synergy_state(x, x_prev)  # [B,1,1,1]
        sigma_next_adapt = torch.clamp((sigma_i*s_in).view(B,1,1,1) / (1.0 + S), min=sigma_min)
        sigma_j = torch.minimum(sigma_next_adapt, (sigma_next_sched*s_in).view(B,1,1,1))

        # deterministic projection to next step
        alpha_j = 1.0 / torch.sqrt(1.0 + sigma_j*sigma_j)
        x0_hat  = denoised  # consistent with 'denoised' returned by model call
        x_new   = alpha_j * x0_hat + sigma_j * eps

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': x0_hat})

        x_prev = x.detach()
        x = x_new.to(dtype).detach()

    return x

# =========================
# ComfyUI Node returning a SAMPLER
# =========================

class GOD_Sampler_Advanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = "MilitantAI/Switchblade/Generation"  # your category

    def build(self):
        # Wrap our function as a KSAMPLER so SamplerCustom/KSampler accept it.  :contentReference[oaicite:3]{index=3}
        sampler = comfy.samplers.KSAMPLER(god_flow_sample)
        return (sampler,)

NODE_CLASS_MAPPINGS = {
    "GOD Sampler (Advanced)": GOD_Sampler_Advanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GOD Sampler (Advanced)": "GOD Sampler (Advanced)",
}
