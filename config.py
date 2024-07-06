import os
from typing import List, Tuple

# Image generation settings
RESOLUTIONS: List[int] = [512, 1024]
NUM_IMAGES_LIST: List[int] = [9, 1]
INFERENCE_STEPS_LIST: List[int] = [4, 10]
DEFAULT_TEMPERATURE: float = 1.0

# File paths
IMAGE_FOLDER: str = "generated_images"
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Model names
MODEL_MID_RES: str = "PixArt-alpha/PixArt-XL-2-512x512"
MODEL_HIGH_RES: str = "PixArt-alpha/PixArt-XL-2-1024-MS"

# User input options
VALID_USER_INPUTS: set = {
    "regenerate", "reselect", "stop", "continue", 
    "prompt", "temperature", "restart"
}

# Initial prompt
INITIAL_PROMPT: str = """
A magical landscape with a towering, crystalline mountain and a serene lake 
reflecting iridescent clouds. At the lake's edge, a glowing blue-leaved tree 
stands amidst bioluminescent plants. A bridge of light arches over the lake 
towards an enchanted castle nestled in the mountain peaks.
"""
