# ComfyUI custom nodes: GOD Scheduler (SIGMAS)
# Drop this file into ComfyUI/custom_nodes/ and restart.

import torch
import torch.nn.functional as F

# =========================
# GOD-S sigmas (Advanced)
# =========================

def _baseline_sigmas(model, steps, device, denoise):
    """
    Build a baseline descending sigma vector (high->low) that honors `steps` and `denoise`.
    We keep this latent-agnostic (Advanced path requires precomputed SIGMAS), while the sampler
    will apply state-adaptive refinement per step.
    """
    # Prefer model-provided range when available (e.g., SDXL in ComfyUI models)
    sigma_max = float(getattr(model, "sigma_max", 14.0))
    sigma_min = float(getattr(model, "sigma_min", 0.01))
    # Karras-like monotone schedule in log space, truncated by denoise fraction
    ks = torch.linspace(0.0, 1.0, steps, device=device)
    # log interpolation (smooth)
    log_hi = torch.log(torch.tensor(sigma_max, device=device))
    log_lo = torch.log(torch.tensor(sigma_min, device=device))
    sigmas_full = torch.exp(log_hi + (log_lo - log_hi) * ks)  # high->low
    # Apply denoise fraction: travel only a fraction of the path
    # Implement by linearly mixing start with each target
    if denoise < 1.0:
        # compress toward start (sigma_max) so end stays higher
        sigmas_full = sigma_max * (sigmas_full / sigma_max).pow(denoise)
    # ensure strictly descending (first is max)
    if sigmas_full[0] < sigmas_full[-1]:
        sigmas_full = torch.flip(sigmas_full, dims=[0])
    return sigmas_full

# =========================
# ComfyUI Nodes
# =========================

class GOD_Scheduler_Advanced:
    """
    Produces SIGMAS for Custom KSampler (Advanced).
    Inputs: model, steps, denoise.
    Output: sigmas (descending).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "steps": ("INT", {"default": 28, "min": 1, "max": 2000}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "make"
    CATEGORY = "MilitantAI/Switchblade/Generation"

    def make(self, model, steps, denoise):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sigmas = _baseline_sigmas(model, steps, device, float(denoise))
        return (sigmas,)

# Required node mappings for ComfyUI to discover them
NODE_CLASS_MAPPINGS = {
    "GOD Scheduler (Advanced)": GOD_Scheduler_Advanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GOD Scheduler (Advanced)": "GOD Scheduler (Advanced)",
}
