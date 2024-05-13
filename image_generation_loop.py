from generate_image import generate_images
from display_image import display_and_select_image

def image_generation_loop(initial_prompt):
    prompt = initial_prompt
    temperature = 0.7  # Default temperature
    temperatures = []  # To store temperature for each iteration
    selected_images = []
    generated_image_sets = []
    resolutions = [256, 512, 1024]
    num_images_list = [3, 2, 1]
    iteration = 0

    while iteration < len(resolutions):
        resolution = resolutions[iteration]
        num_images = num_images_list[iteration]
        base_image = selected_images[-1] if selected_images and iteration > 0 else None

        # Use the specific temperature for the current iteration or the default if not set
        current_temperature = temperatures[iteration] if iteration < len(temperatures) else temperature
        print(f"Current prompt: {prompt}")
        print(f"Current temperature for this iteration: {current_temperature}")

        # Generate images if they have not been generated or need regeneration
        generated_images = generate_images(prompt, num_images=num_images, resolution=resolution, temp=current_temperature, base_image=base_image)
        generated_image_sets.append(generated_images) if iteration >= len(generated_image_sets) else generated_image_sets[iteration]

        selected_image = display_and_select_image(generated_images, resolution, iteration)

        while True:  # User input loop
            user_input = input("Options: type 'regenerate' to recreate images, 'restart' to start over, 'reselect' to choose previous image again, 'stop' to exit, 'prompt' to change prompt, or 'temperature' to change temperature: ").strip().lower()

            if user_input == "regenerate":
                # Regenerate images using current settings but do not advance the iteration
                generated_images = generate_images(prompt, num_images=num_images, resolution=resolution, temp=current_temperature, base_image=base_image)
                generated_image_sets[iteration] = generated_images
                selected_image = display_and_select_image(generated_images, resolution, iteration)
                continue
            elif user_input == "restart":
                selected_images = []
                generated_image_sets = []
                temperatures = []
                iteration = 0
                break
            elif user_input == "reselect":
                iteration = max(0, iteration - 1)
                break
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
            else:
                print("Invalid option. Please enter a valid command.")
                continue

            break

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

        iteration += 1  # Move to the next iteration only when ready

    return selected_images
