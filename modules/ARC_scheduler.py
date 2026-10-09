import math
import torch
import comfy.samplers


class ARC_Scheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "MilitantAI/Switchblade/Generation"
    FUNCTION = "get_sigmas"

    # --- Baseline (match GOD Scheduler Advanced behaviour) ---
    @staticmethod
    def _baseline_sigmas(model, steps: int, denoise: float) -> torch.Tensor:
        """Log-space descending schedule using model sigma bounds, with denoise exponent.

        - steps: number of steps (kept constant)
        - denoise in [0,1]: 1.0 = full path; <1 compresses toward sigma_max
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sigma_max = float(getattr(model, "sigma_max", 14.0))
        sigma_min = float(getattr(model, "sigma_min", 0.01))

        ks = torch.linspace(0.0, 1.0, steps, device=device)
        log_hi = torch.log(torch.tensor(sigma_max, device=device))
        log_lo = torch.log(torch.tensor(sigma_min, device=device))
        sigmas = torch.exp(log_hi + (log_lo - log_hi) * ks)  # high->low

        if denoise < 1.0:
            sigmas = sigma_max * (sigmas / sigma_max).pow(denoise)

        if sigmas[0] < sigmas[-1]:
            sigmas = torch.flip(sigmas, dims=[0])
        return sigmas

    # --- Deterministic ARC warp helpers ---
    @staticmethod
    def _logistic(p: float, k: float = 6.0) -> float:
        """Centered logistic at p=0.5 with temperature k."""
        return 1.0 / (1.0 + math.exp(-k * (p - 0.5)))

    @classmethod
    def _normalized_logistic(cls, p: float, k: float = 6.0) -> float:
        """Map logistic(p) to [0,1] regardless of k."""
        l0 = cls._logistic(0.0, k)
        l1 = cls._logistic(1.0, k)
        lp = cls._logistic(p, k)
        # Numerical safety
        denom = max(1e-8, (l1 - l0))
        return (lp - l0) / denom

    @classmethod
    def _compute_arc_warp(cls, base_sigmas: torch.Tensor) -> torch.Tensor:
        """Compute a deterministic, gentle ARC warp over progress in [0,1].

        - No external parameters; amplitude adapts to dynamic range.
        - Endpoints preserved (warp=1 at p=0 and p=1).
        - Monotonicity enforced without clamping to fixed floors/ceilings.
        """
        steps = len(base_sigmas)
        if steps <= 1:
            return base_sigmas.clone()

        # Dynamic-range-aware amplitude (deterministic)
        sigma_start = float(base_sigmas[0])
        sigma_end = float(base_sigmas[-1])
        ratio = max(1.0, sigma_start / max(1e-12, sigma_end))
        # Map ratio to a modest amplitude in [0.12, 0.32]
        amplitude = 0.12 + min(0.20, 0.05 * math.log10(ratio))
        k = 6.0  # logistic temperature

        # Build warp factors across progress (anchored at endpoints)
        warp = []
        for i in range(steps):
            p = i / (steps - 1)
            u01 = cls._normalized_logistic(p, k)  # in [0,1]
            signed = 2.0 * u01 - 1.0  # in [-1,1]
            anchor = 4.0 * p * (1.0 - p)  # 0 at ends, peak at mid
            warp.append(1.0 + amplitude * signed * anchor)
        warp = torch.tensor(warp, dtype=base_sigmas.dtype, device=base_sigmas.device)

        # Apply warp
        adjusted = base_sigmas * warp

        # Enforce strict monotonic decrease
        for i in range(1, steps):
            if adjusted[i] >= adjusted[i - 1]:
                adjusted[i] = torch.nextafter(adjusted[i - 1], torch.tensor(0.0, dtype=adjusted.dtype, device=adjusted.device))

        return adjusted

    def get_sigmas(self, model, steps, denoise):
        """Generate a deterministic ARC-warped baseline schedule.

        UI options: steps, denoise (0.0..1.0). The model defines sigma bounds.
        """
        denoise = float(max(0.0, min(1.0, denoise)))

        # Baseline in log space with denoise exponent (keeps steps constant)
        base_sigmas = self._baseline_sigmas(model, int(steps), denoise)

        # Deterministic ARC warp (self-driven)
        adjusted_sigmas = self._compute_arc_warp(base_sigmas)

        return (adjusted_sigmas,)



# =========================
# Stock KSampler scheduler registration
# =========================

ARC_SCHEDULER_NAME = "arc"

def arc_scheduler(model_sampling, steps: int) -> torch.Tensor:
    """
    Stock-KSampler adapter for ARC.

    Contract:
      - takes (model_sampling, steps)
      - returns 1-D sigmas of length steps + 1
      - final sigma is exactly 0

    Stock KSampler already handles denoise by requesting a longer schedule
    and slicing it, so this adapter always builds the full ARC path.
    """
    steps = max(1, int(steps))

    base_sigmas = ARC_Scheduler._baseline_sigmas(
        model_sampling,
        steps,
        1.0,
    )
    adjusted_sigmas = ARC_Scheduler._compute_arc_warp(base_sigmas)

    return torch.cat(
        [adjusted_sigmas, adjusted_sigmas.new_zeros([1])],
        dim=0,
    )


def _register_arc_scheduler():
    """Register ARC in ComfyUI's live scheduler tables/lists without core edits."""
    comfy.samplers.SCHEDULER_HANDLERS[ARC_SCHEDULER_NAME] = (
        comfy.samplers.SchedulerHandler(arc_scheduler, use_ms=True)
    )

    names = getattr(comfy.samplers, "SCHEDULER_NAMES", None)
    if isinstance(names, list) and ARC_SCHEDULER_NAME not in names:
        names.append(ARC_SCHEDULER_NAME)

    schedulers = getattr(comfy.samplers.KSampler, "SCHEDULERS", None)
    if isinstance(schedulers, list) and ARC_SCHEDULER_NAME not in schedulers:
        schedulers.append(ARC_SCHEDULER_NAME)

_register_arc_scheduler()

# Register the node into ComfyUI's NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "ARC Scheduler": ARC_Scheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ARC Scheduler": "ARC Scheduler",
}
