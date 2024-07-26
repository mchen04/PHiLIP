# image_generation_loop.py

import os
import shutil
import logging
from typing import List, Optional
import numpy as np
from PIL import Image
from textwrap import dedent
from generate_image import generate_images
from display_image import display_and_select_image, save_images
from user_input_handler import handle_user_input, get_user_input
from image_enhancement import apply_freestyle, latent_upscale_image, apply_controlnet
from config import (
    IMAGE_FOLDER, RESOLUTIONS, NUM_IMAGES_LIST, 
    INFERENCE_STEPS_LIST, DEFAULT_TEMPERATURE,
    LOG_FORMAT, LOG_DATE_FORMAT, TEMPERATURE_PROMPT,
    INFERENCE_STEPS_PROMPT, NUM_IMAGES_PROMPT,
    ENHANCEMENT_PROMPT, ENHANCEMENT_OPTIONS
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

def clear_generated_images_folder() -> None:
    """Clears all files in the generated_images folder."""
    if os.path.exists(IMAGE_FOLDER):
        shutil.rmtree(IMAGE_FOLDER)
    os.makedirs(IMAGE_FOLDER)

def image_generation_loop(initial_prompt: str) -> Optional[List[np.ndarray]]:
    """
    Main loop for image generation process.
    
    Args:
        initial_prompt: Initial prompt for image generation.
    
    Returns:
        Optional[List[np.ndarray]]: List of final selected images or None if process is stopped.
    """
    clear_generated_images_folder()

    prompt = initial_prompt
    temperature = DEFAULT_TEMPERATURE
    resolution = RESOLUTIONS[0]  # Start with 512x512
    num_images = NUM_IMAGES_LIST[0]
    inference_steps = INFERENCE_STEPS_LIST[0]

    while True:
        logger.info(dedent(f"""
        Current settings:
        Prompt: {prompt}
        Temperature: {temperature}
        Resolution: {resolution}
        Inference steps: {inference_steps}
        Number of images: {num_images}
        """))

        generated_images = generate_images(prompt, num_images, resolution, temperature, None, inference_steps)

        selected_images = display_and_select_image(generated_images, resolution, 0)
        if not selected_images:
            logger.warning("No images selected. Exiting program.")
            return None

        enhancement_option = get_user_input(ENHANCEMENT_PROMPT, str, valid_options=ENHANCEMENT_OPTIONS)
        
        if enhancement_option == "Freestyle":
            enhanced_images = [apply_freestyle(img, prompt) for img in selected_images]
        elif enhancement_option == "Upscaler":
            enhanced_images = [latent_upscale_image(Image.fromarray(img), prompt) for img in selected_images]
        elif enhancement_option == "ControlNet":
            enhanced_images = [apply_controlnet(img, prompt) for img in selected_images]
        elif enhancement_option == "Pixart":
            enhanced_images = generate_images(prompt, 1, 1024, temperature, selected_images, INFERENCE_STEPS_LIST[1])
        else:  # None
            enhanced_images = selected_images

        # Save the final enhanced images
        final_resolution = 1024 if enhancement_option in ["Upscaler", "Pixart"] else resolution
        save_images(enhanced_images, final_resolution, final=True)
        logger.info(f"Final enhanced image(s) saved to {IMAGE_FOLDER}")

        user_action = handle_user_input()
        if user_action == "stop":
            logger.info("User requested to stop. Exiting program.")
            return enhanced_images
        elif user_action == "regenerate":
            logger.info("Regenerating images...")
            clear_generated_images_folder()
            continue
        elif user_action == "change_temp":
            temperature = get_user_input(TEMPERATURE_PROMPT, float, 0.5, 1.5)
        elif user_action == "change_prompt":
            prompt = input("Enter new prompt: ")
        elif user_action == "change_steps":
            inference_steps = get_user_input(INFERENCE_STEPS_PROMPT, int, 1, 100)
        elif user_action == "change_num_images":
            num_images = get_user_input(NUM_IMAGES_PROMPT, int, 1, 9)
        elif user_action == "continue":
            return enhanced_images

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    result = image_generation_loop("Test prompt")
    print(f"Final result: {result}")
