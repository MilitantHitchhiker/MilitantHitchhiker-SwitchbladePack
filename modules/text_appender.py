import folder_paths
import os

class TextAppender_v2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "text1": ("STRING", {"default": ""}),
                "text2": ("STRING", {"default": ""}),
                "text3": ("STRING", {"default": ""}),
                "text4": ("STRING", {"default": ""}),
                "text5": ("STRING", {"default": ""}),
                "delimiter": ("STRING", {"default": "\\n"}),
                "output_file": ("STRING", {"default": "none"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "append_text"
    OUTPUT_NODE = True
    CATEGORY = "Switchblade"

    def append_text(self, text1="", text2="", text3="", text4="", text5="", delimiter="\\n", output_file="none"):
        texts = [text for text in [text1, text2, text3, text4, text5] if text]
        if not texts:
            print("No text to append.")
            return ("",)
        
        delimiter = delimiter.replace("\\n", "\n")
        result = delimiter.join(texts)

        if output_file != "none":
            try:
                output_dir = folder_paths.get_output_directory()
                file_path = os.path.join(output_dir, output_file)
                
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(result + delimiter)
                print(f"Successfully appended text to {output_file}")
            except Exception as e:
                print(f"Error appending to file {output_file}: {str(e)}")
        else:
            print("No output file specified. Text only returned as output.")

        return (result,)

NODE_CLASS_MAPPINGS = {
    "TextAppender_v2": TextAppender_v2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextAppender_v2": "Text Appender v2"
}
