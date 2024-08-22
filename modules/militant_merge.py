import torch
import json
from collections import defaultdict
import comfy.model_management as model_management
import random
import logging

class MilitantMergeNodev2:
    def __init__(self):
        self.device = model_management.get_torch_device()
        self.setup_logger()

    def setup_logger(self):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model1": ("MODEL",),
                "model2": ("MODEL",),
                "use_smaller_model": ("BOOLEAN", {"default": False}),
                "merge_mode": (["simple", "dare"],),
                "input_blocks_merge_amount": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "middle_block_merge_amount": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "output_blocks_merge_amount": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "force_keep_dim": ("BOOLEAN", {"default": False}),
                "random_drop_probability": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "mask_model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    FUNCTION = "flux_dare_analyze_merge"
    CATEGORY = "Switchblade"

    def get_model_size(self, model):
        # Attempt to get size from model metadata
        if hasattr(model, 'metadata') and 'file_size' in model.metadata:
            return model.metadata['file_size'] / (1024 * 1024 * 1024)  # Convert bytes to GB
        # If metadata is not available, estimate size based on parameters
        param_size = sum(p.numel() * p.element_size() for p in model.model.parameters())
        estimated_size = param_size / (1024 * 1024 * 1024)  # Convert bytes to GB
        self.logger.warning(f"Model size not found in metadata. Estimated size: {estimated_size:.2f} GB")
        return estimated_size

    def flux_dare_analyze_merge(self, model1, model2, use_smaller_model, merge_mode,
                                input_blocks_merge_amount, middle_block_merge_amount, output_blocks_merge_amount,
                                force_keep_dim, random_drop_probability, mask_model=None):
        size1 = self.get_model_size(model1)
        size2 = self.get_model_size(model2)
        self.logger.info(f"Model 1 size: {size1:.2f} GB")
        self.logger.info(f"Model 2 size: {size2:.2f} GB")
        self.logger.info(f"Model 1 parameter count: {sum(p.numel() for p in model1.model.parameters())}")
        self.logger.info(f"Model 2 parameter count: {sum(p.numel() for p in model2.model.parameters())}")
        self.logger.info(f"Use smaller model: {use_smaller_model}")

        if use_smaller_model:
            base_model, secondary_model = (model1, model2) if size1 <= size2 else (model2, model1)
            self.logger.info(f"Selected {'Model 1' if size1 <= size2 else 'Model 2'} as base (smaller) model")
        else:
            base_model, secondary_model = (model1, model2) if size1 >= size2 else (model2, model1)
            self.logger.info(f"Selected {'Model 1' if size1 >= size2 else 'Model 2'} as base (larger) model")

        m1 = base_model.clone()
        m2 = secondary_model

        # Log model attributes for debugging
        self.logger.debug(f"Base model attributes: {dir(m1)}")
        self.logger.debug(f"Secondary model attributes: {dir(m2)}")

        merge_info = self.perform_block_merge(m1, m2, merge_mode,
                                              input_blocks_merge_amount, middle_block_merge_amount, output_blocks_merge_amount,
                                              force_keep_dim, random_drop_probability, mask_model)

        final_size = self.get_model_size(m1)
        self.logger.info(f"Final model size: {final_size:.2f} GB")

        analysis_summary = self.generate_minimal_summary(size1, size2, final_size, use_smaller_model, merge_info)
        return (m1, analysis_summary)

    def merge_tensors(self, tp1, tp2, amount, mode, force_keep_dim, random_drop_probability, mask=None):
        if tp1.shape != tp2.shape:
            if force_keep_dim:
                return tp1
            else:
                raise ValueError(f"Tensor shapes do not match: {tp1.shape} vs {tp2.shape}")

        if mode == "dare":
            # DARE merge implementation
            abs_diff = torch.abs(tp1 - tp2)
            max_diff = torch.max(abs_diff)
            if max_diff > 0:
                importance = abs_diff / max_diff
            else:
                importance = torch.ones_like(abs_diff)
            rand_tensor = torch.rand_like(tp1)
            merge_mask = (rand_tensor > random_drop_probability) & (rand_tensor < importance + random_drop_probability)
            merged = torch.where(merge_mask,
                                tp1 * (1 - amount) + tp2 * amount,
                                tp1)
        else:
            # Simple linear interpolation
            merged = tp1 * (1 - amount) + tp2 * amount

        if mask is not None:
            merged = merged * mask + tp1 * (1 - mask)

        return merged


    def get_block_from_key(self, key):
        if 'input_blocks' in key or 'down_blocks' in key:
            return 'input_blocks'
        elif 'middle_block' in key:
            return 'middle_block'
        elif 'output_blocks' in key or 'up_blocks' in key:
            return 'output_blocks'
        return "other"

    def perform_block_merge(self, m1, m2, merge_mode,
                            input_blocks_merge_amount, middle_block_merge_amount, output_blocks_merge_amount,
                            force_keep_dim, random_drop_probability, mask_model):
        merge_info = {
            "merge_mode": merge_mode,
            "input_blocks_merge_amount": input_blocks_merge_amount,
            "middle_block_merge_amount": middle_block_merge_amount,
            "output_blocks_merge_amount": output_blocks_merge_amount,
            "force_keep_dim": force_keep_dim,
            "random_drop_probability": random_drop_probability,
            "components": defaultdict(dict),
            "errors": []
        }

        block_merge_amounts = {
            "input_blocks": input_blocks_merge_amount,
            "middle_block": middle_block_merge_amount,
            "output_blocks": output_blocks_merge_amount,
            "other": 0.5  # Default merge amount for other blocks
        }

        total_params = 0
        merged_params = 0
        kept_params = 0
        error_params = 0

        for name, param in m1.model.named_parameters():
            total_params += 1
            self.logger.debug(f"Processing parameter: {name}")
            if name in m2.model.state_dict():
                block = self.get_block_from_key(name)
                t1 = param.data
                t2 = m2.model.state_dict()[name].data
                mask = mask_model.model.state_dict()[name].data if mask_model and name in mask_model.model.state_dict() else None

                try:
                    self.logger.debug(f"Shapes - t1: {t1.shape}, t2: {t2.shape}")
                    self.logger.debug(f"Types - t1: {t1.dtype}, t2: {t2.dtype}")
                    merged_tensor = self.merge_tensors(t1, t2, block_merge_amounts[block], merge_mode,
                                                       force_keep_dim, random_drop_probability, mask)
                    if not torch.allclose(merged_tensor, t1, rtol=1e-05, atol=1e-08):
                        param.data.copy_(merged_tensor)
                        merge_info["components"][name] = self.get_merge_component_info(block_merge_amounts[block], t1, t2, merged_tensor)
                        self.logger.debug(f"Successfully merged: {name}")
                        merged_params += 1
                    else:
                        merge_info["components"][name] = "Kept from base model (no significant change after merge)"
                        self.logger.debug(f"No significant change after merge: {name}")
                        kept_params += 1
                except Exception as e:
                    self.logger.error(f"Error merging {name}: {str(e)}")
                    merge_info["errors"].append(f"Error merging {name}: {str(e)}")
                    merge_info["components"][name] = "Kept from base model due to merge error"
                    error_params += 1
            else:
                self.logger.debug(f"Parameter not found in secondary model: {name}")
                merge_info["components"][name] = "Kept from base model (not present in secondary model)"
                kept_params += 1

        merge_info["components_merged"] = merged_params
        merge_info["components_kept"] = kept_params
        merge_info["components_errored"] = error_params

        self.logger.info(f"Merge complete. Total parameters: {total_params}")
        self.logger.info(f"Components merged: {merged_params}, kept: {kept_params}, errors: {error_params}")

        return merge_info

    def generate_minimal_summary(self, size1, size2, final_size, use_smaller_model, merge_info):
        base_size = min(size1, size2) if use_smaller_model else max(size1, size2)
        summary = {
            "model_sizes": {
                "model1": f"{size1:.2f}B",
                "model2": f"{size2:.2f}B",
                "final": f"{final_size:.2f}B"
            },
            "merge_stats": {
                "components_merged": merge_info["components_merged"],
                "components_kept": merge_info["components_kept"],
                "errors": len(merge_info["errors"])
            },
            "settings": {
                "use_smaller_model": use_smaller_model,
                "merge_mode": merge_info["merge_mode"],
                "merge_amounts": [
                    merge_info["input_blocks_merge_amount"],
                    merge_info["middle_block_merge_amount"],
                    merge_info["output_blocks_merge_amount"]
                ]
            }
        }
        if abs(final_size - base_size) / base_size > 0.01:  # Allow 1% tolerance
            summary["warning"] = "Unexpected final model size"
        return json.dumps(summary, indent=2)

    def get_merge_component_info(self, merge_amount, t1, t2, merged_tensor):
        return {
            "merge_amount": merge_amount,
            "original_mean": float(t1.mean()),
            "original_std": float(t1.std()),
            "secondary_mean": float(t2.mean()),
            "secondary_std": float(t2.std()),
            "merged_mean": float(merged_tensor.mean()),
            "merged_std": float(merged_tensor.std())
        }

NODE_CLASS_MAPPINGS = {
    "MilitantMergeNodev2": MilitantMergeNodev2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MilitantMergeNodev2": "Militant Merge Node (FLUX) v2"
}
