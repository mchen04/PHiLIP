import os
import shutil
from generate_image import generate_images
from display_image import display_and_select_image
from user_input_handler import handle_user_input

def clear_generated_images_folder(folder="generated_images"):
    """Clears all files in the generated_images folder."""
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def image_generation_loop(initial_prompt):
    clear_generated_images_folder()

    prompt = initial_prompt
    temperature = 0.5  # Default temperature
    temperatures = []  # To store temperature for each iteration
    selected_images = []
    generated_image_sets = []
    resolutions = [512, 1024]
    num_images_list = [9, 1]
    inference_steps = [5, 10]  # Default inference steps for each stage
    iteration = 0

    while iteration < len(resolutions):
        resolution = resolutions[iteration]
        num_images = num_images_list[iteration]
        steps = inference_steps[iteration]
        base_images = selected_images if selected_images and iteration > 0 else None

        current_temperature = temperatures[iteration] if iteration < len(temperatures) else temperature
        print(f"Current prompt: {prompt}")
        print(f"Current temperature for this iteration: {current_temperature}")
        print(f"Current inference steps for this iteration: {steps}")

        if iteration >= len(generated_image_sets):
            generated_images = generate_images(prompt, num_images=num_images, resolution=resolution, temp=current_temperature, base_images=base_images, steps=steps)
            generated_image_sets.append(generated_images)
        else:
            generated_images = generated_image_sets[iteration]

        selected_images = display_and_select_image(generated_images, resolution, iteration)

        user_input = handle_user_input()

        if user_input == "regenerate":
            generated_images = generate_images(prompt, num_images=num_images, resolution=resolution, temp=current_temperature, base_images=selected_images, steps=steps)
            generated_image_sets[iteration] = generated_images
            selected_images = display_and_select_image(generated_images, resolution, iteration)
            continue
        elif user_input == "restart":
            selected_images = []
            generated_image_sets = []
            temperatures = []
            iteration = 0
            continue
        elif user_input == "reselect":
            iteration = max(0, iteration - 1)
            continue
        elif user_input == "stop":
            print("User requested to stop. Exiting.")
            return
        elif user_input == "prompt":
            prompt = input("Enter new prompt: ")
            continue
        elif user_input == "temperature":
            try:
                new_temp = float(input("Enter new temperature (suggested range from 0.5 to 1.5): "))
                temperature = new_temp
                if iteration == len(temperatures):
                    temperatures.append(temperature)
                else:
                    temperatures[iteration] = temperature
                current_temperature = temperature
            except ValueError:
                print("Invalid temperature input. Using previous temperature.")
            continue
        elif user_input == "inference steps":
            try:
                new_steps = int(input("Enter new inference steps (positive integer): "))
                inference_steps[iteration] = new_steps
                steps = new_steps
            except ValueError:
                print("Invalid steps input. Using previous steps.")
            continue
        elif user_input == "continue":
            pass

        if selected_images is None:
            print("No images selected, exiting.")
            return

        iteration += 1

    return selected_images
