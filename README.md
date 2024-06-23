# Militant Hitchhiker's Switchblade Pack

The Militant Hitchhiker's Switchblade Pack is a collection of custom nodes for ComfyUI that provide advanced text manipulation and prompt generation functionality. It includes the following nodes:

1. Text Appender
2. Integrated Random Prompt Generator

## Installation

1. Open a terminal or command prompt.
2. Navigate to the `custom_nodes` directory of your ComfyUI installation.
3. Run the following command to clone the repository:
   ```
   git clone https://github.com/MilitantHitchhiker/MilitantHitchhiker-SwitchbladePack.git
   ```
4. Restart ComfyUI to load the new nodes.

## Node Details

### Text Appender

The Text Appender node allows you to concatenate up to five text inputs with a specified delimiter and optionally append the result to a file.

#### Inputs

- `text1` (optional): The first text input. Default is an empty string.
- `text2` (optional): The second text input. Default is `None`.
- `text3` (optional): The third text input. Default is `None`.
- `text4` (optional): The fourth text input. Default is `None`.
- `text5` (optional): The fifth text input. Default is `None`.
- `delimiter` (optional): The delimiter to use for joining the text inputs. Default is `\n` (newline). Use `\\n` in the input field to represent a newline character.
- `output_file` (optional): The name of the file to append the result to. The file will be created in the `dictionaries` folder. Default is an empty string (no file output).

#### Outputs

- `result`: The concatenated text, with the specified delimiter joining the non-empty text inputs.

### Integrated Random Prompt Generator

The Integrated Random Prompt Generator node generates random prompts by selecting items from up to four dictionary files and joining them with a specified delimiter.

#### Inputs

- `dict1_file`, `dict2_file`, `dict3_file`, `dict4_file` (required): The names of the dictionary files to use. The files should be placed in the `input/Dictionaries` folder of your ComfyUI installation. Use `none` to disable a dictionary.
- `enable_dict1`, `enable_dict2`, `enable_dict3`, `enable_dict4` (required): Boolean values indicating whether to enable each dictionary. Default is `True` for all dictionaries.
- `dict1_delimiter`, `dict2_delimiter`, `dict3_delimiter`, `dict4_delimiter` (required): The delimiters used to split the items in each dictionary file. Default is `\n` (newline) for all dictionaries.
- `output_delimiter` (required): The delimiter used to join the selected items from each dictionary. Default is a space.
- `seed` (required): The random seed for reproducibility. Default is `0`.

#### Outputs

- `result`: The generated prompt, with items randomly selected from the enabled dictionaries and joined by the `output_delimiter`.

## Usage Examples

### Text Appender

1. Connect the Text Appender node to a text input node.
2. Set the `delimiter` to `\\n` to join the text inputs with newline characters.
3. Specify the `output_file` to append the result to a file.
4. Run the graph to concatenate the text inputs and save the result to the specified file.

### Integrated Random Prompt Generator

1. Place the dictionary files you want to use in the `input/Dictionaries` folder of your ComfyUI installation.
2. Connect the Integrated Random Prompt Generator node to other nodes as desired.
3. Select the dictionary files you want to use and set the corresponding `enable_dict` inputs to `True`.
4. Specify the delimiters for each dictionary file using the `dict_delimiter` inputs.
5. Set the `output_delimiter` to control how the selected items are joined.
6. Specify a `seed` value for reproducibility, or leave it as `0` for random results.
7. Run the graph to generate random prompts based on the selected dictionaries.

## Contributing

Contributions to the Militant Hitchhiker's Switchblade Pack are welcome! If you have any ideas, bug reports, or feature requests, please open an issue on the [GitHub repository](https://github.com/MilitantHitchhiker/MilitantHitchhiker-SwitchbladePack). Pull requests are also encouraged.

## License

The Militant Hitchhiker's Switchblade Pack is released under the [MIT License](https://opensource.org/licenses/MIT).
