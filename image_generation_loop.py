import os
import shutil
import logging
import numpy as np
from typing import List, Optional
from generate_image import generate_images
from display_image import display_and_select_image
from user_input_handler import handle_user_input
from config import IMAGE_FOLDER, RESOLUTIONS, NUM_IMAGES_LIST, INFERENCE_STEPS_LIST, DEFAULT_TEMPERATURE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clear_generated_images_folder() -> None:
    """Clears all files in the generated_images folder."""
    if os.path.exists(IMAGE_FOLDER):
        shutil.rmtree(IMAGE_FOLDER)
    os.makedirs(IMAGE_FOLDER)

def image_generation_loop(initial_prompt: str) -> Optional[List[np.ndarray]]:
    """
    Main loop for image generation process.
    
    Args:
        initial_prompt (str): Initial prompt for image generation.
    
    Returns:
        Optional[List[np.ndarray]]: List of final selected images or None if process is stopped.
    """
    clear_generated_images_folder()

    prompt = initial_prompt
    temperature = DEFAULT_TEMPERATURE
    temperatures: List[float] = []
    selected_images: List[np.ndarray] = []
    generated_image_sets: List[List[np.ndarray]] = []
    iteration = 0

    while iteration < len(RESOLUTIONS):
        resolution = RESOLUTIONS[iteration]
        num_images = NUM_IMAGES_LIST[iteration]
        inference_steps = INFERENCE_STEPS_LIST[iteration]
        
        base_images = selected_images if selected_images and iteration > 0 else None
        current_temperature = temperatures[iteration] if iteration < len(temperatures) else temperature

        print_current_settings(prompt, current_temperature, resolution, inference_steps)

        generated_images = generate_images(prompt, num_images, resolution, current_temperature, base_images, inference_steps)
        generated_image_sets.append(generated_images)

        selected_image = display_and_select_image(generated_images, resolution, iteration)

        user_input = handle_user_input()

        if user_input == "regenerate":
            continue
        elif user_input == "restart":
            selected_images, generated_image_sets, temperatures, iteration = [], [], [], 0
            continue
        elif user_input == "reselect":
            iteration = max(0, iteration - 1)
            continue
        elif user_input == "stop":
            logging.info("User requested to stop. Exiting.")
            return None
        elif user_input == "prompt":
            prompt = input("Enter new prompt: ")
            continue
        elif user_input == "temperature":
            temperature = get_new_temperature()
            if iteration == len(temperatures):
                temperatures.append(temperature)
            else:
                temperatures[iteration] = temperature
            continue
        elif user_input == "continue":
            pass

        if selected_image is None:
            logging.warning("No image selected, exiting.")
            return None

        selected_images.extend(selected_image if isinstance(selected_image, list) else [selected_image])

        if iteration >= len(temperatures):
            temperatures.append(current_temperature)

        iteration += 1

    return selected_images

def print_current_settings(prompt: str, temperature: float, resolution: int, inference_steps: int) -> None:
    """
    Print current settings for image generation.
    
    Args:
        prompt (str): Current prompt.
        temperature (float): Current temperature.
        resolution (int): Current resolution.
        inference_steps (int): Current number of inference steps.
    """
    logging.info(f"Current prompt: {prompt}")
    logging.info(f"Current temperature: {temperature}")
    logging.info(f"Current resolution: {resolution}")
    logging.info(f"Inference steps: {inference_steps}")

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
            logging.warning("Invalid temperature input. Please enter a valid number.")
