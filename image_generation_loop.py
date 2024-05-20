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
    # Clear the generated_images folder at the start
    clear_generated_images_folder()

    prompt = initial_prompt
    temperature = 1.0  # Default temperature
    temperatures = []  # To store temperature for each iteration
    selected_images = []
    generated_image_sets = []
    resolutions = [256, 512, 1024]
    num_images_list = [3, 2, 1]
    mask_fractions = [0.4, 0.3, 0.1]  # Progressive masking fractions
    iteration = 0

    while iteration < len(resolutions):
        resolution = resolutions[iteration]
        num_images = num_images_list[iteration]
        mask_fraction = mask_fractions[iteration]
        base_image = selected_images[-1] if selected_images and iteration > 0 else None

        # Use the specific temperature for the current iteration or the default if not set
        current_temperature = temperatures[iteration] if iteration < len(temperatures) else temperature
        print(f"Current prompt: {prompt}")
        print(f"Current temperature for this iteration: {current_temperature}")

        # Generate images if they have not been generated or need regeneration
        if iteration >= len(generated_image_sets):
            generated_images = generate_images(prompt, num_images=num_images, resolution=resolution, temp=current_temperature, base_image=base_image, mask_fraction=mask_fraction)
            generated_image_sets.append(generated_images)
        else:
            generated_images = generated_image_sets[iteration]

        selected_image = display_and_select_image(generated_images, resolution, iteration)

        user_input = handle_user_input()

        if user_input == "regenerate":
            generated_images = generate_images(prompt, num_images=num_images, resolution=resolution, temp=current_temperature, base_image=base_image, mask_fraction=mask_fraction)
            generated_image_sets[iteration] = generated_images
            selected_image = display_and_select_image(generated_images, resolution, iteration)
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
        elif user_input == "continue":
            pass  # Exit the loop to move to the next iteration

        if selected_image is None:
            print("No image selected, exiting.")
            return

        # Manage the selection and temperatures for each iteration
        if len(selected_images) > iteration:
            selected_images[iteration] = selected_image
        else:
            selected_images.append(selected_image)

        if iteration >= len(temperatures):
            temperatures.append(current_temperature)  # Ensure each iteration's temperature is recorded

        iteration += 1  # Move to the next iteration only when ready or 'continue' is selected

    return selected_images
