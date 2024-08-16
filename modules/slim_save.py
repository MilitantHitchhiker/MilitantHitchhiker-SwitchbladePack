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
        state_dict = model.state_dict()
        save_file(state_dict, f"{filename_prefix}.safetensors")
        return ()

# This line is needed for ComfyUI to recognize and load the custom node
NODE_CLASS_MAPPINGS = {
    "ModelWeightsSave": ModelWeightsSave
}

# This line is optional, but can be used to specify a user-friendly name for the node
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelWeightsSave": "Save Model Weights"
}