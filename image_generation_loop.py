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

def image_generation_loop():
    while True:
        clear_generated_images_folder()

        prompt = input("Enter a prompt for image generation: ").strip()
        
        while True:
            try:
                temperature = float(input("Enter the temperature for image generation (suggested range from 0.5 to 1.5): ").strip())
                if 0.5 <= temperature <= 1.5:
                    break
                else:
                    print("Please enter a temperature within the suggested range from 0.5 to 1.5.")
            except ValueError:
                print("Invalid input. Please enter a valid number for temperature.")

        selected_images = []
        resolutions = [512, 1024]
        num_images_list = [9, 1]
        inference_steps = [5, 10] 
        iteration = 0

        while iteration < len(resolutions):
            resolution = resolutions[iteration]
            num_images = num_images_list[iteration]
            steps = inference_steps[iteration]
            base_images = selected_images if selected_images and iteration > 0 else None

            print(f"Generating images for prompt: {prompt}")
            print(f"Temperature for this iteration: {temperature}")

            generated_images = generate_images(prompt, num_images=num_images, resolution=resolution, temp=temperature, base_images=base_images, steps=steps)

            selected_images = display_and_select_image(generated_images, resolution, iteration)

            if selected_images is None:
                print("No images selected. Restarting with a new prompt and temperature.")
                break

            if iteration == 1:  # When at 1024 resolution, provide limited options
                user_input = input("Options: type 'regenerate' to recreate images, 'restart' to start over, or 'stop' to exit: ").strip().lower()
                if user_input == "regenerate":
                    continue
                elif user_input == "restart":
                    break
                elif user_input == "stop":
                    print("User requested to stop. Exiting.")
                    return
            else:
                user_input = handle_user_input()

                if user_input == "regenerate":
                    continue
                elif user_input == "reselect":
                    iteration = max(0, iteration - 1)
                    continue
                elif user_input == "stop":
                    print("User requested to stop. Exiting.")
                    return
                elif user_input == "continue":
                    pass

            iteration += 1

        if iteration == len(resolutions):
            print("Image generation completed successfully.")
            return selected_images
