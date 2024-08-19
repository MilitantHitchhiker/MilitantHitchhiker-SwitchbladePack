"""
@author: Militant Hitchhiker
@title: Switchblade Pack
@nickname: Switchblade
@description: Militant Hitchhiker's multi-function nodes.
"""

import folder_paths
import os
import sys
import importlib

comfy_path = os.path.dirname(folder_paths.__file__)
switchblade_path = os.path.dirname(__file__)
modules_path = os.path.join(switchblade_path, "modules")
sys.path.append(modules_path)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

print(f"### Loading: Militant Hitchhiker's Switchblade Pack' (Switchblade v1)")

# Module initialisation, modules located in the ./modules directory. Append using filename.
NODE_MODULES = [
    "random_prompt_generator",
    "text_appender",
    "flux_save",
]

def load_nodes(module_name: str):
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    module = importlib.import_module(module_name, package=__name__)
    # Merge the imported module's NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS with the global ones
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

def write_nodes_list(module_names: list[str]):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(this_dir, "nodes.log")
    lines = []
    for module_name in module_names:
        module = importlib.import_module(module_name, package=__name__)
        lines.append(module_name.strip("."))
        for identifier, display_name in module.NODE_DISPLAY_NAME_MAPPINGS.items():
            lines.append(f" {identifier}: {display_name}")
        lines.append("")
    with open(path, "w", encoding="utf8") as f:
        f.write("\n".join(lines))

# Dynamically load all nodes specified in NODE_MODULES
for module_name in NODE_MODULES:
    load_nodes(module_name)