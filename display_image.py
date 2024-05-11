import matplotlib.pyplot as plt
import numpy as np
import os

def display_and_select_image(images, resolution):
    """Displays images, prompts the user to select their favorite, and saves all images."""
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images > 1:
        for i, img in enumerate(images):
            axs[i].imshow(img if isinstance(img, np.ndarray) else np.array(img))
            axs[i].axis('off')
            axs[i].set_title(f"Image {i+1}")
    else:
        axs.imshow(images[0] if isinstance(images[0], np.ndarray) else np.array(images[0]))
        axs.axis('off')
        axs.set_title("Image 1")
    plt.tight_layout()
    plt.show()

    image_folder = "generated_images"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # Save all images with generic names
    for i, img in enumerate(images):
        file_path = os.path.join(image_folder, f"image_{resolution}px_{i+1}.png")
        plt.imsave(file_path, img if isinstance(img, np.ndarray) else np.array(img))
        print(f"Image {i+1} saved as {file_path}")

    # Ask user for their favorite image
    while True:
        choice = input(f"Select your favorite image (1-{num_images}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= num_images:
            favorite_index = int(choice) - 1
            favorite_image = images[favorite_index]
            new_path = os.path.join(image_folder, f"image_{resolution}px_selected.png")
            os.rename(os.path.join(image_folder, f"image_{resolution}px_{choice}.png"), new_path)
            print(f"Your favorite image {choice} has been renamed to 'image_{resolution}px_selected.png'")
            return favorite_image
        else:
            print(f"Invalid input. Please enter a number between 1 and {num_images}.")