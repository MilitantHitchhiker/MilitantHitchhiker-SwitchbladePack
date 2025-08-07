import json
import os
import subprocess
import sys
import stat

try:
    from groq import Groq, APITimeoutError, APIConnectionError
except ImportError:
    print("Installing groq library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "groq"])
    from groq import Groq, APITimeoutError, APIConnectionError

class GroqAPIPromptEnhancer:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(__file__), "groq_config.json")
        self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Create a config with default values
            self.config = {"api_key": "", "system_prompt": ""}
            self.save_config()
            print(f"groq_config.json created with default values. Please edit at: {self.config_path}")
        except json.JSONDecodeError:
            print(f"Error decoding groq_config.json. Please ensure it is valid JSON at: {self.config_path}")
            # Reset to default values
            self.config = {"api_key": "", "system_prompt": ""}
            self.save_config()

    def save_config(self):
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=4)
        os.chmod(self.config_path, stat.S_IRUSR | stat.S_IWUSR)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["gemma2-9b-it", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"],),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "override_system_prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_text",)
    FUNCTION = "execute"
    CATEGORY = "MilitantAI/Switchblade/Text Processing"

    def execute(self, model, text, override_system_prompt=None):
        if not self.config.get("api_key"):
            return ("Error: API key not configured. Please edit groq_config.json",)

        # Correctly handle the default system prompt
        system_prompt_to_use = override_system_prompt if override_system_prompt else self.config.get("system_prompt", "")

        client = Groq(api_key=self.config["api_key"])

        messages = []
        if system_prompt_to_use:
            messages.append({"role": "system", "content": system_prompt_to_use})
        messages.append({"role": "user", "content": text})

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                stop=None,
                stream=False
            )
            return (completion.choices[0].message.content,)

        except APITimeoutError as e:
            return (f"Groq API Timeout Error: {e}",)
        except APIConnectionError as e:
            return (f"Groq API Connection Error: {e}",)
        except Exception as e:
            return (f"Groq API Error: {e}",)

NODE_CLASS_MAPPINGS = {
    "GroqAPIPromptEnhancer": GroqAPIPromptEnhancer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroqAPIPromptEnhancer": "Groq API Prompt Enhancer"
}