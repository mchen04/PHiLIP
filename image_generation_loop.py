# image_generation_loop.py

import os
import shutil
import logging
from typing import List, Optional
import numpy as np
from textwrap import dedent
from generate_image import generate_images, upscale_image
from display_image import display_and_select_image
from user_input_handler import handle_user_input
from config import (
    IMAGE_FOLDER, DEFAULT_TEMPERATURE,
    LOG_FORMAT, LOG_DATE_FORMAT
)
from PIL import Image

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

def clear_generated_images_folder() -> None:
    """Clears all files in the generated_images folder."""
    if os.path.exists(IMAGE_FOLDER):
        shutil.rmtree(IMAGE_FOLDER)
    os.makedirs(IMAGE_FOLDER)

def image_generation_loop(initial_prompt: str) -> Optional[np.ndarray]:
    """
    Main loop for image generation process.
    
    Args:
        initial_prompt: Initial prompt for image generation.
    
    Returns:
        Optional[np.ndarray]: Final upscaled image as a numpy array or None if process is stopped.
    """
    clear_generated_images_folder()

    prompt = initial_prompt
    temperature = DEFAULT_TEMPERATURE
    resolution = 512
    num_images = 9
    inference_steps = 6
    
    while True:
        logger.info(dedent(f"""
        Current settings:
        Prompt: {prompt}
        Temperature: {temperature}
        Resolution: {resolution}
        Inference steps: {inference_steps}
        """))

        generated_images = generate_images(prompt, num_images, resolution, temperature, None, inference_steps)

        selected_image = display_and_select_image(generated_images, resolution, 0)

        if selected_image is None:
            logger.warning("No image selected, exiting.")
            return None

        user_input = handle_user_input()

        if user_input == "regenerate":
            continue
        elif user_input == "restart":
            return None
        elif user_input == "reselect":
            continue
        elif user_input == "stop":
            logger.info("User requested to stop. Exiting.")
            return None
        elif user_input == "prompt":
            prompt = input("Enter new prompt: ")
            continue
        elif user_input == "temperature":
            temperature = get_new_temperature()
            continue
        elif user_input == "continue":
            break

    logger.info("Upscaling the selected image to 1024x1024...")
    upscaled_image = upscale_image(selected_image, prompt, output_size=(1024, 1024))
    return upscaled_image

def get_new_temperature() -> float:
    """
    Get new temperature from user input.
    
    Returns:
        float: New temperature value.
    """
    while True:
        try:
            new_temp = float(input("Enter new temperature (suggested range from 0.5 to 1.5): "))
            return new_temp
        except ValueError:
            logger.warning("Invalid temperature input. Please enter a valid number.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    result = image_generation_loop("Test prompt")
    print(f"Final result: {result}")
