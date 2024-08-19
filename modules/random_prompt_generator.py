import os
import sys
import random

class IntegratedRandomPromptGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        # Get the path of the current script
        script_path = os.path.abspath(__file__)
        # Navigate up the directory tree to reach the "ComfyUI" folder
        comfyui_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path))))
        # Construct the path to the "Dictionaries" folder
        dictionaries_folder = os.path.join(comfyui_path, "input", "Dictionaries")
        files = cls.get_all_txt_files(dictionaries_folder)
        return {
            "required": {
                "dict1_file": (["none"] + files,),
                "dict2_file": (["none"] + files,),
                "dict3_file": (["none"] + files,),
                "dict4_file": (["none"] + files,),
                "enable_dict1": ("BOOLEAN", {"default": True}),
                "enable_dict2": ("BOOLEAN", {"default": True}),
                "enable_dict3": ("BOOLEAN", {"default": True}),
                "enable_dict4": ("BOOLEAN", {"default": True}),
                "dict1_delimiter": ("STRING", {"default": "\n"}),
                "dict2_delimiter": ("STRING", {"default": "\n"}),
                "dict3_delimiter": ("STRING", {"default": "\n"}),
                "dict4_delimiter": ("STRING", {"default": "\n"}),
                "output_delimiter": ("STRING", {"default": " "}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    @staticmethod
    def get_all_txt_files(directory):
        txt_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):
                    # Use relative path from the dictionaries folder
                    relative_path = os.path.relpath(os.path.join(root, file), directory)
                    txt_files.append(relative_path)
        return sorted(txt_files)

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "Switchblade"

    def generate(self, dict1_file, dict2_file, dict3_file, dict4_file,
                 enable_dict1, enable_dict2, enable_dict3, enable_dict4,
                 dict1_delimiter, dict2_delimiter, dict3_delimiter, dict4_delimiter,
                 output_delimiter, seed):
        random.seed(seed)
       
        dictionaries_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "input", "Dictionaries")
       
        def load_and_process_dict(file, delimiter, enabled):
            if not enabled or file == "none":
                return []
            try:
                with open(os.path.join(dictionaries_folder, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                return [item.strip() for item in content.split(delimiter) if item.strip()]
            except Exception as e:
                print(f"Error reading file {file}: {str(e)}")
                return []

        dictionaries = [
            load_and_process_dict(dict1_file, dict1_delimiter, enable_dict1),
            load_and_process_dict(dict2_file, dict2_delimiter, enable_dict2),
            load_and_process_dict(dict3_file, dict3_delimiter, enable_dict3),
            load_and_process_dict(dict4_file, dict4_delimiter, enable_dict4),
        ]

        selected_items = []
        for dictionary in dictionaries:
            if dictionary:
                selected_items.append(random.choice(dictionary))

        result = output_delimiter.join(selected_items)
        print(f"Generated result: {result}")  # Debug print
        return (result,)

NODE_CLASS_MAPPINGS = {
    "IntegratedRandomPromptGenerator": IntegratedRandomPromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntegratedRandomPromptGenerator": "Integrated Random Prompt Generator"
}