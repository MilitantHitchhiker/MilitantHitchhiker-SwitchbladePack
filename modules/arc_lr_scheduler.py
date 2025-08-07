import math
import torch
from comfy.samplers import calculate_sigmas  # For generating initial sigma schedule

class GODARCScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 1e-6, "max": 1.0, "step": 1e-4}),
                "sigma_max": ("FLOAT", {"default": 14.614642, "min": 1e-1, "max": 100.0, "step": 1e-1}),
                "rho": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "gravity_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "recursive_depth": ("INT", {"default": 3, "min": 1, "max": 10}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "MilitantAI/GOD Framework/Schedulers"
    FUNCTION = "get_sigmas"

    def __init__(self):
        self.iterations = 0

    def gravity_fn(self, progress, strength):
        """Gravity-like function to scale sigmas adaptively."""
        return 1 / (1 + math.exp(-strength * progress))

    def recursive_adjustment(self, sigma, gravity_fn, depth, progress, strength):
        """Apply recursive adjustments to sigma."""
        if depth == 0:
            return sigma
        adjusted_sigma = gravity_fn(progress, strength) * sigma
        return self.recursive_adjustment(adjusted_sigma, gravity_fn, depth - 1, progress, strength)

    def apply_arc_to_sigmas(self, base_sigmas, sigma_min, sigma_max, gravity_strength, recursive_depth):
        """Apply GOD-ARC logic to adjust sigmas."""
        adjusted_sigmas = []
        steps = len(base_sigmas)
        for i, sigma in enumerate(base_sigmas):
            progress = i / (steps - 1)
            adjusted_sigma = self.recursive_adjustment(sigma, self.gravity_fn, recursive_depth, progress, gravity_strength)
            # Clamp adjusted sigma to valid range
            adjusted_sigma = max(sigma_min, min(adjusted_sigma, sigma_max))
            adjusted_sigmas.append(adjusted_sigma)
        return torch.tensor(adjusted_sigmas)

    def get_sigmas(self, model, steps, sigma_min, sigma_max, rho, gravity_strength, recursive_depth):
        """Generate GOD-ARC adjusted sigmas."""
        # Generate base sigmas using calculate_sigmas
        base_sigmas = calculate_sigmas(model.get_model_object("model_sampling"), "karras", steps).cpu()

        # Apply GOD-ARC logic
        adjusted_sigmas = self.apply_arc_to_sigmas(
            base_sigmas, sigma_min, sigma_max, gravity_strength, recursive_depth
        )
        return (adjusted_sigmas,)


# Register the node into ComfyUI's NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "GODARCScheduler": GODARCScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GODARCScheduler": "GOD ARC Scheduler",
}
