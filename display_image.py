import matplotlib.pyplot as plt
import numpy as np
import os

def display_and_select_image(images, resolution, iteration):
    """Displays images, prompts the user to select their favorite, and saves all images."""
    num_images = len(images)
    if num_images == 1:
        fig, axs = plt.subplots(1, num_images, figsize=(5, 5))
        axs = [axs]  # Make axs a list even if there is only one subplot
    else:
        fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    for i, img in enumerate(images):
        axs[i].imshow(img if isinstance(img, np.ndarray) else np.array(img))
        axs[i].axis('off')
        axs[i].set_title(f"Image {i+1}")

    plt.tight_layout()
    plt.show()

    image_folder = "generated_images"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    for i, img in enumerate(images):
        file_path = os.path.join(image_folder, f"{resolution}-{i+1}.png")
        plt.imsave(file_path, img if isinstance(img, np.ndarray) else np.array(img))
        print(f"Image {i+1} saved as {file_path}")

    selected_images = []
    while True:
        choice = input(f"Select your favorite images (1-{num_images}), separated by commas, or type 'stop' to exit: ").strip()

        if choice.lower() == 'stop':
            return None
        else:
            try:
                selected_indices = [int(x) - 1 for x in choice.split(',')]
                selected_images = [images[i] for i in selected_indices if 0 <= i < num_images]
                for i in selected_indices:
                    new_path = os.path.join(image_folder, f"{resolution}-selected-{i+1}.png")
                    os.rename(os.path.join(image_folder, f"{resolution}-{i+1}.png"), new_path)
                    print(f"Your favorite image {i+1} has been renamed to '{resolution}-selected-{i+1}.png'")
                return selected_images
            except ValueError:
                print(f"Invalid input. Please enter numbers between 1 and {num_images} separated by commas or 'stop' to exit.")
