from generate_image import generate_images
from display_image import display_and_select_image

def image_generation_loop(prompt):
    selected_images = []
    generated_image_sets = []
    resolutions = [256, 512, 1024]
    num_images_list = [3, 2, 1]
    iteration = 0

    while iteration < len(resolutions):
        resolution = resolutions[iteration]
        num_images = num_images_list[iteration]
        base_image = selected_images[-1] if selected_images else None

        if not generated_image_sets or len(generated_image_sets) <= iteration:
            generated_images = generate_images(prompt, num_images=num_images, resolution=resolution, temp=1.0, base_image=base_image)
            generated_image_sets.append(generated_images)
        else:
            generated_images = generated_image_sets[iteration]

        selected_image = display_and_select_image(generated_images, resolution, iteration)

        if selected_image == "regenerate":
            generated_images = generate_images(prompt, num_images=num_images, resolution=resolution, temp=1.0, base_image=base_image)
            generated_image_sets[iteration] = generated_images
            continue
        elif selected_image == "restart":
            selected_images = []
            generated_image_sets = []
            iteration = 0
            continue
        elif selected_image == "reselect":
            iteration = max(0, iteration - 1)
            continue
        elif selected_image == "stop":
            print("User requested to stop. Exiting.")
            return
        elif selected_image is None:
            print("No image selected, exiting.")
            return

        if len(selected_images) > iteration:
            selected_images[iteration] = selected_image
        else:
            selected_images.append(selected_image)

        iteration += 1

    return selected_images
