import torch
import json
from collections import defaultdict

class ModelAnalyserNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyse_model"
    CATEGORY = "Switchblade"

    def analyse_model(self, model):
        model_state_dict = model.model.state_dict()
        analysis = {}

        # Model Structure Analysis
        analysis["structure"] = self.analysis_structure(model_state_dict)

        # Model Size Analysis
        analysis["size_info"] = self.analysis_size(model_state_dict)

        # Block Analysis
        analysis["block_info"] = self.analysis_blocks(model_state_dict)

        # Data Type Analysis
        analysis["dtype_info"] = self.analysis_dtypes(model_state_dict)

        # Additional Information
        analysis["additional_info"] = {
            "total_tensors": len(model_state_dict),
            "total_size_gb": sum(t.numel() * t.element_size() for t in model_state_dict.values()) / (1024**3)
        }

        return (json.dumps(analysis, indent=2),)

    def analysis_structure(self, state_dict):
        structure = defaultdict(int)
        for key in state_dict.keys():
            top_level = key.split('.')[0]
            structure[top_level] += 1
        return dict(structure)

    def analysis_size(self, state_dict):
        size_info = {
            "total_size_gb": sum(t.numel() * t.element_size() for t in state_dict.values()) / (1024**3),
            "largest_tensors": []
        }
        sorted_tensors = sorted(state_dict.items(), key=lambda x: x[1].numel(), reverse=True)
        size_info["largest_tensors"] = [
            {"name": name, "shape": list(tensor.shape), "size_mb": tensor.numel() * tensor.element_size() / (1024**2)}
            for name, tensor in sorted_tensors[:5]
        ]
        return size_info

    def analysis_blocks(self, state_dict):
        block_info = {
            "double_blocks": len([k for k in state_dict.keys() if k.startswith('double_blocks')]),
            "single_blocks": len([k for k in state_dict.keys() if k.startswith('single_blocks')]),
            "other_blocks": len([k for k in state_dict.keys() if not (k.startswith('double_blocks') or k.startswith('single_blocks'))])
        }
        return block_info

    def analysis_dtypes(self, state_dict):
        dtype_counts = defaultdict(int)
        for tensor in state_dict.values():
            dtype_counts[str(tensor.dtype)] += 1
        return dict(dtype_counts)

NODE_CLASS_MAPPINGS = {
    "ModelAnalyserNode": ModelAnalyserNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelAnalyserNode": "Model Analyser"
}
