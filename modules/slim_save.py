import torch
from safetensors.torch import save_file
import os
import folder_paths

class ModelWeightsSave:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "model_weights"})
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_weights"
    OUTPUT_NODE = True
    CATEGORY = "advanced/model_saving"

    def save_weights(self, model, filename_prefix):
        if hasattr(model, 'model'):  # Check if it's a ModelPatcher object
            state_dict = model.model.state_dict()
        elif hasattr(model, 'state_dict'):  # Check if it's a regular PyTorch model
            state_dict = model.state_dict()
        else:
            raise ValueError("Unsupported model type. Cannot extract state_dict.")
        
        # Filter out non-tensor items from the state dict
        filtered_state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        
        # Get the output directory
        output_dir = folder_paths.get_output_directory()
        
        # Create the full path for the output file
        output_path = os.path.join(output_dir, f"{filename_prefix}.safetensors")
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the file
        save_file(filtered_state_dict, output_path)
        print(f"Model weights saved to {output_path}")
        return ()

# This line is needed for ComfyUI to recognize and load the custom node
NODE_CLASS_MAPPINGS = {
    "ModelWeightsSave": ModelWeightsSave
}

# This line is optional, but can be used to specify a user-friendly name for the node
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelWeightsSave": "Save Model Weights"
}