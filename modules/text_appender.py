import folder_paths
import os

class TextAppender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "text1": ("STRING", {"default": ""}),
                "text2": ("STRING", {"default": ""}),
                "text3": ("STRING", {"default": ""}),
                "text4": ("STRING", {"default": ""}),
                "text5": ("STRING", {"default": ""}),
                "input_delimiter": ("STRING", {"default": "\\n"}),  
                "output_delimiter": ("STRING", {"default": "\\n"}),  
                "output_file": ("STRING", {"default": "none"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "append_text"
    OUTPUT_NODE = True
    CATEGORY = "MilitantAI/Switchblade/Text Processing"

    def append_text(self, text1="", text2="", text3="", text4="", text5="", input_delimiter="", output_delimiter="", output_file="none"):
        # Collect non-empty texts
        texts = [text for text in [text1, text2, text3, text4, text5] if text]
        if not texts:
            print("No text to append.")
            return ("",)
        
        # If the delimiters are specified as \n, convert to actual newline, otherwise leave empty delimiter as is
        input_delimiter = input_delimiter.replace("\\n", "\n") if input_delimiter else ""
        output_delimiter = output_delimiter.replace("\\n", "\n") if output_delimiter else ""
        
        # Join texts with input_delimiter
        result = input_delimiter.join(texts)
        
        # Append output_delimiter at the end only if specified
        if output_delimiter:
            result += output_delimiter

        if output_file != "none":
            try:
                # Get output directory
                output_dir = folder_paths.get_output_directory()
                
                # Ensure output directory exists, create if not
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Create full file path
                file_path = os.path.join(output_dir, output_file)
                
                # Open the file in append mode, create it if it doesn't exist
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(result)
                print(f"Successfully appended text to {file_path}")
            
            except Exception as e:
                print(f"Error appending to file {file_path}: {str(e)}")
        else:
            print("No output file specified. Text only returned as output.")

        return (result,)


NODE_CLASS_MAPPINGS = {
    "TextAppender_v2": TextAppender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextAppender_v2": "Text Appender"
}
