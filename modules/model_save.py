import torch
from safetensors.torch import save_file
import os
import json
import folder_paths

class ModelSave_v2:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "flux_model"}),
                "output_format": (["bfloat16", "float16", "float32", "int8"], {"default": "bfloat16"})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_flux_model"
    OUTPUT_NODE = True
    CATEGORY = "Switchblade"

    def save_flux_model(self, model, filename_prefix, output_format):
        state_dict = model.model.state_dict()
        flux_state_dict = {}
        metadata = {"format": "flux", "model_type": "FLUX", "dtype": output_format}

        print(f"Original state_dict keys: {state_dict.keys()}")
        print(f"Original state_dict size: {self.get_state_dict_size(state_dict):.2f} GB")

        for key, value in state_dict.items():
            if key.startswith('diffusion_model.'):
                new_key = key.replace('diffusion_model.', '', 1)
                flux_state_dict[new_key] = self.convert_tensor(value, output_format)
            else:
                flux_state_dict[key] = self.convert_tensor(value, output_format)

        print(f"Filtered flux_state_dict keys: {flux_state_dict.keys()}")
        print(f"Filtered flux_state_dict size: {self.get_state_dict_size(flux_state_dict):.2f} GB")

        filename = f"{filename_prefix}.safetensors"
        output_path = os.path.join(self.output_dir, filename)
        save_file(flux_state_dict, output_path, metadata=metadata)
        
        file_size = os.path.getsize(output_path)
        
        print(f"Flux model saved to {output_path}")
        print(f"Total tensors saved: {len(flux_state_dict)}")
        print(f"Saved file size: {file_size / (1024**3):.2f} GB")

        self.save_tensor_info(flux_state_dict, output_path)

        return {}

    def convert_tensor(self, tensor, output_format):
        if output_format == "bfloat16":
            return tensor.bfloat16()
        elif output_format == "float16":
            return tensor.half()
        elif output_format == "float32":
            return tensor.float()
        elif output_format == "int8":
            return tensor.to(torch.int8)
        else:
            return tensor  # Keep original format if not specified

    def get_state_dict_size(self, state_dict):
        return sum(t.numel() * t.element_size() for t in state_dict.values()) / (1024**3)

    def save_tensor_info(self, state_dict, output_path):
        tensor_info = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "size_gb": v.numel() * v.element_size() / (1024**3)} for k, v in state_dict.items()}
        info_filename = f"{os.path.splitext(output_path)[0]}_info.json"
        with open(info_filename, "w") as f:
            json.dump(tensor_info, f, indent=2)

NODE_CLASS_MAPPINGS = {
    "FluxModelSave_v2": ModelSave_v2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxModelSave_v2": "Save Flux Model v2"
}
