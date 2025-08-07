import os
import logging
import comfy.utils
import json
from collections import defaultdict
from comfy.sd import load_diffusion_model_state_dict

# Define the custom node class
class UniLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Dynamically list available models in the "models/unet/" directory
        unet_dir = os.path.join("models", "unet")
        models_dir = os.path.join("models", "checkpoints")
        unet_files = [f for f in os.listdir(unet_dir) if f.endswith((".safetensors", ".ckpt"))]
        model_files = [f for f in os.listdir(models_dir) if f.endswith((".safetensors", ".ckpt"))]
        model_list = unet_files + model_files

        return {
            "required": {
                "unet_name": (model_list, {"default": model_list[0] if model_list else ""}),  # Dropdown with model names
            }
        }
    
    RETURN_TYPES = ("MODEL", "STRING")  # Outputs: Loaded model and filename
    FUNCTION = "load_custom_unet"
    CATEGORY = "Custom Nodes/Model Loaders"  # Category in ComfyUI
    CATEGORY = "MilitantAI/Switchblade/Model Merging"

    json_output = ""

    def analyse_model(self, model):
        model_state_dict = model.model.state_dict()
        analyse = {}

        # Model Structure Analysis
        analyse["structure"] = self.analyse_structure(model_state_dict)
        # Additional Information
        analyse["additional_info"] = {
            "total_tensors": len(model_state_dict),
            "total_size_gb": sum(t.numel() * t.element_size() for t in model_state_dict.values()) / (1024**3)
        }

        # Model Size Analysis
        analyse["size_info"] = self.analyse_size(model_state_dict)

        # Block Analysis
        analyse["block_info"] = self.analyse_unique_block_sizes(model_state_dict)
        analyse["block_patterns"] = self.analyse_unique_block_patterns(model_state_dict)

        # Data Type Analysis
        analyse["dtype_info"] = self.analyse_dtypes(model_state_dict)

        # VAE and CLIP Analysis
        analyse["vae_clip_info"] = self.analyse_vae_clip(model_state_dict)


        json_output = json.dumps(analyse, indent=2)
        return (json_output,)

    def analyse_structure(self, state_dict):
        structure = defaultdict(int)
        for key in state_dict.keys():
            top_level = key.split('.')[0]
            structure[top_level] += 1
        return dict(structure)

    def analyse_size(self, state_dict):
        size_info = {
            "total_size_gb": sum(t.numel() * t.element_size() for t in state_dict.values()) / (1024**3),
            "largest_tensors": []
        }
        sorted_tensors = sorted(state_dict.items(), key=lambda x: x[1].numel(), reverse=True)
        size_info["largest_tensors"] = [
            {"name": name, "shape": list(tensor.shape), "size_mb": tensor.numel() * tensor.element_size() / (1024**2)}
            for name, tensor in sorted_tensors #[:5]
        ]
        return size_info

    def analyse_unique_block_sizes(self, state_dict):
        block_sizes = defaultdict(list)

        for name, tensor in state_dict.items():
            shape_tuple = tuple(tensor.shape)  # Convert shape to tuple for uniqueness
            block_sizes[shape_tuple].append(name)  # Group tensors by shape

        # Convert defaultdict to regular dictionary for JSON serialization
        unique_blocks = {str(shape): names for shape, names in block_sizes.items()}
        
        return unique_blocks
    
    def analyse_unique_block_patterns(self, state_dict):
        block_patterns = defaultdict(set)

        for name in state_dict.keys():
            prefix = '.'.join(name.split('.')[:2])  # Group by first two segments
            block_patterns[prefix].add(name)

        return {key: list(value) for key, value in block_patterns.items()}


    def analyse_dtypes(self, state_dict):
        dtype_counts = defaultdict(int)
        for tensor in state_dict.values():
            dtype_counts[str(tensor.dtype)] += 1
        return dict(dtype_counts)


    def analyse_vae_clip(self, state_dict):
        vae_keywords = ["vae", "decoder", "encoder", "bottleneck"]
        clip_keywords = ["clip", "text_model", "vision_model", "transformer"]

        vae_tensors = []
        clip_tensors = []

        for name in state_dict.keys():
            if any(keyword in name.lower() for keyword in vae_keywords):
                vae_tensors.append(name)
            if any(keyword in name.lower() for keyword in clip_keywords):
                clip_tensors.append(name)

        analyse_extras = {
            "has_vae": len(vae_tensors) > 0,
            "has_clip": len(clip_tensors) > 0,
            "vae_tensors": vae_tensors,
            "clip_tensors": clip_tensors
        }

        return analyse_extras




    def load_custom_unet(self, unet_path, d_type="float32", model_options={}):
        # Load the U-Net model using the provided path
        try:
            sd = comfy.utils.load_torch_file(unet_path)
            model = load_diffusion_model_state_dict(sd, model_options=model_options)
            if model is None:
                logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
                raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
        except Exception as e:
            logging.error(f"Failed to load U-Net: {e}")
            raise

        # Convert the model to the specified data type (precision)
        if d_type == "float16":
            model = model.half()  # Convert to float16
        elif d_type == "float32":
            model = model.float()  # Convert to float32
        else:
            raise ValueError(f"Unsupported d_type: {d_type}")

        # Extract the filename from the path
        unet_filename = os.path.basename(unet_path)

        # Return both the loaded model and its filename
        return (model, unet_filename)
