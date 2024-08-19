import os

class TextAppender:
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
            },
            "optional": {
                "text1": ("STRING", {"default": ""}),
                "text2": ("STRING", {"default": None}),
                "text3": ("STRING", {"default": None}),
                "text4": ("STRING", {"default": None}),
                "text5": ("STRING", {"default": None}),
                "delimiter": ("STRING", {"default": "\\n"}),
                "output_file": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "append_text"
    OUTPUT_NODE = True
    CATEGORY = "Switchblade"

    @staticmethod
    def get_all_txt_files(directory):
        txt_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):
                    relative_path = os.path.relpath(os.path.join(root, file), directory)
                    txt_files.append(relative_path)
        return sorted(txt_files)

    def append_text(self, text1, text2=None, text3=None, text4=None, text5=None, delimiter="\n", output_file=""):
        # Collect all non-empty texts
        texts = [text for text in [text1, text2, text3, text4, text5] if text]

        if not texts:
            print("No text to append.")
            return ("",)
        
        # Handle the special character sequence "\n" in the delimiter
        delimiter = delimiter.replace("\\n", "\n")

        result = delimiter.join(texts)

        if output_file:
            dictionaries_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "input", "Dictionaries")
            file_path = os.path.join(dictionaries_folder, output_file)

            try:
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(result + delimiter)
                print(f"Successfully appended text to {output_file}")
            except Exception as e:
                print(f"Error appending to file {output_file}: {str(e)}")
        else:
            print("No output file specified. Text only returned as output.")

        return (result,)

NODE_CLASS_MAPPINGS = {
    "TextAppender": TextAppender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextAppender": "Text Appender"
}