import torch
from safetensors.torch import save_file

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
        
        save_file(filtered_state_dict, f"{filename_prefix}.safetensors")
        print(f"Model weights saved to {filename_prefix}.safetensors")
        return ()

# This line is needed for ComfyUI to recognize and load the custom node
NODE_CLASS_MAPPINGS = {
    "ModelWeightsSave": ModelWeightsSave
}

# This line is optional, but can be used to specify a user-friendly name for the node
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelWeightsSave": "Save Model Weights"
}