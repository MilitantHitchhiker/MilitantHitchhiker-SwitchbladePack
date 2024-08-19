import os
import torch
import folder_paths
from safetensors.torch import save_file

class FluxModelSaveNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "filename": ("STRING", {"default": "flux_model_weights.safetensors"})
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_flux_model"
    OUTPUT_NODE = True
    CATEGORY = "Switchblade"

    def save_flux_model(self, model, filename):
        state_dict = model.model.state_dict()
        flux_state_dict = {}

        top_level_keys = [
            'double_blocks', 'final_layer', 'guidance_in', 'img_in',
            'single_blocks', 'time_in', 'txt_in', 'vector_in'
        ]

        for key, value in state_dict.items():
            if key.startswith('diffusion_model.'):
                new_key = key.replace('diffusion_model.', '', 1)
                if any(new_key.startswith(prefix) for prefix in top_level_keys):
                    flux_state_dict[new_key] = value
            elif any(key.startswith(prefix) for prefix in top_level_keys):
                flux_state_dict[key] = value

        output_path = os.path.join(self.output_dir, filename)
        save_file(flux_state_dict, output_path)
        
        print(f"Flux model saved to {output_path}")
        print(f"Total tensors saved: {len(flux_state_dict)}")
        return {}

NODE_CLASS_MAPPINGS = {
    "FluxModelSave": FluxModelSaveNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxModelSave": "Save Flux Model"
}
