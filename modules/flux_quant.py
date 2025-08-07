import torch
import json
import os
import folder_paths
from collections import defaultdict
from safetensors.torch import save_file

class FluxQuantNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "precision": (["auto", "float32", "float16", "bfloat16", "float8_e5m2", "float8_e4m3fn"], {"default": "auto"}),
                #"file_name": ("STRING", {"default": "model_quant"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyse_and_save_model"
    CATEGORY = "MilitantAI/Switchblade/Model Merging"

    def analyse_and_save_model(self, model, precision):
        try:
            device, chosen_precision = self.get_device_and_precision(precision)

            # Load the model state dictionary from input model
            model_state_dict = model.model.state_dict()
            print(f"Loaded model with {len(model_state_dict)} tensors.")

            # Convert tensors to selected precision
            model_state_dict = self.convert_to_selected_precision(model_state_dict, chosen_precision, device)

            # Perform analysis
            analysis = {
                "structure": self.analyse_structure(model_state_dict),
                "size_info": self.analyse_size(model_state_dict),
                "block_info": self.analyse_blocks(model_state_dict),
                "dtype_info": self.analyse_dtypes(model_state_dict),
                "additional_info": {
                    "total_tensors": len(model_state_dict),
                    "total_size_gb": sum(t.numel() * t.element_size() for t in model_state_dict.values()) / (1024**3)
                }
            }

            
            file_name = model.ckpt_name if hasattr(model, 'ckpt_name') else "unknown_model"
            file_name = f"{file_name}_{chosen_precision}.safetensors"


            # Save the processed model
            self.save_model(model_state_dict, file_name, chosen_precision)

            return (json.dumps(analysis, indent=2), f"Model saved as {file_name}")

        except Exception as e:
            return (f"Error processing model: {str(e)}", "")

    def get_device_and_precision(self, user_selection):
        """Determine the best device and user-selected precision."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float8_e5m2": torch.float8_e5m2 if hasattr(torch, "float8_e5m2") else torch.float16,  # Fallback
            "float8_e4m3fn": torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16,  # Fallback
        }

        if user_selection == "auto":
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()[0]
                if capability >= 8:  # Ada/Hopper support FP8
                    chosen_precision = torch.float8_e5m2
                else:
                    chosen_precision = torch.float16
            else:
                chosen_precision = torch.float32
        else:
            chosen_precision = precision_map[user_selection]

        print(f"Using device: {device}, precision: {chosen_precision}")
        return device, chosen_precision

    def convert_to_selected_precision(self, state_dict, precision, device):
        """Convert all tensors to user-selected precision."""
        print(f"Converting tensors to {precision}...")
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(precision).to(device, non_blocking=True)
        torch.cuda.empty_cache()
        print("All tensors converted successfully.")
        return state_dict

    def save_model(self, model_state_dict, file_name, chosen_precision):
        """Save the model in the selected precision format."""
        try:
            # Get output directory
            output_dir = folder_paths.get_output_directory()
            
            # Ensure output directory exists, create if not
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Create full file path
            output_path = os.path.join(output_dir, file_name)

            output_dir = os.path.dirname("output_path")
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Move tensors to CPU before saving to avoid GPU memory issues
            state_dict_cpu = {k: v.cpu() for k, v in model_state_dict.items()}
            
            save_file(state_dict_cpu, output_path)
            print(f"Model successfully saved to {output_path} in {chosen_precision} precision.")
        except Exception as e:
            print(f"Error saving model: {e}")

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
            {
                "name": name, 
                "shape": list(tensor.shape), 
                "size_mb": tensor.numel() * tensor.element_size() / (1024**2)
            }
            for name, tensor in sorted_tensors[:5]
        ]
        return size_info

    def analyse_blocks(self, state_dict):
        block_info = {
            "double_blocks": len([k for k in state_dict.keys() if k.startswith('double_blocks')]),
            "single_blocks": len([k for k in state_dict.keys() if k.startswith('single_blocks')]),
            "other_blocks": len([k for k in state_dict.keys() if not (k.startswith('double_blocks') or k.startswith('single_blocks'))])
        }
        return block_info

    def analyse_dtypes(self, state_dict):
        dtype_counts = defaultdict(int)
        for tensor in state_dict.values():
            dtype_counts[str(tensor.dtype)] += 1
        return dict(dtype_counts)

# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxQuantNode": FluxQuantNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxQuantNode": "Flux Quant Node"
}
