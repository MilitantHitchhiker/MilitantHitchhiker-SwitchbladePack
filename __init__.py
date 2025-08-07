"""
@author: Militant Hitchhiker
@title: Switchblade Pack
@nickname: Switchblade
@description: Militant Hitchhiker's multi-function nodes.
"""

import os
import sys
import importlib
import logging
from typing import Dict, Any

import folder_paths

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path setup
comfy_path = os.path.dirname(folder_paths.__file__)
switchblade_path = os.path.dirname(__file__)
modules_path = os.path.join(switchblade_path, "modules")
sys.path.append(modules_path)

# Node mappings
NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

# Module configuration
NODE_MODULES = [
    "random_prompt_generator",
    "text_appender",
    #"model_save",
    "model_analyser",
    "flux_quant",
    #"arc_lr_scheduler",
    "groq_node",
]

def load_nodes(module_name: str) -> None:
    """Load node classes and display names from a module."""
    try:
        module = importlib.import_module(module_name, package=__name__)
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        logger.info(f"Loaded module: {module_name}")
    except ImportError as e:
        logger.error(f"Error loading module {module_name}: {e}")

def write_nodes_list(module_names: list[str]) -> None:
    """Write the list of loaded nodes to a log file."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(this_dir, "nodes.log")
    lines = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name, package=__name__)
            lines.append(module_name.strip("."))
            for identifier, display_name in module.NODE_DISPLAY_NAME_MAPPINGS.items():
                lines.append(f" {identifier}: {display_name}")
            lines.append("")
        except ImportError as e:
            lines.append(f"Error loading {module_name}: {e}")
    
    with open(path, "w", encoding="utf8") as f:
        f.write("\n".join(lines))

def initialize_switchblade() -> None:
    """Initialize the Switchblade Pack."""
    logger.info("Loading: Militant Hitchhiker's Switchblade Pack (Switchblade v1.3)")
    
    for module_name in NODE_MODULES:
        load_nodes(module_name)
    
    write_nodes_list(NODE_MODULES)

# Initialize the pack
initialize_switchblade()
