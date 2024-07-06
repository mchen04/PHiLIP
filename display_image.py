import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from typing import List, Optional, Union
from config import IMAGE_FOLDER

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def display_and_select_image(images: List[np.ndarray], resolution: int, iteration: int) -> Optional[List[np.ndarray]]:
    """
    Displays images, prompts the user to select their favorite, and saves all images.
    
    Args:
        images (List[np.ndarray]): List of images to display and save.
        resolution (int): Resolution of the images.
        iteration (int): Current iteration number.
    
    Returns:
        Optional[List[np.ndarray]]: List of selected images or None if no selection is made.
    """
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    axs = axs if num_images > 1 else [axs]

    for i, img in enumerate(images):
        axs[i].imshow(img if isinstance(img, np.ndarray) else np.array(img))
        axs[i].axis('off')
        axs[i].set_title(f"Image {i+1}")

    plt.tight_layout()
    plt.show()

    save_images(images, resolution)

    return get_user_selection(images, num_images, resolution)

def save_images(images: List[np.ndarray], resolution: int) -> None:
    """
    Save all generated images.
    
    Args:
        images (List[np.ndarray]): List of images to save.
        resolution (int): Resolution of the images.
    """
    for i, img in enumerate(images):
        file_path = os.path.join(IMAGE_FOLDER, f"{resolution}-{i+1}.png")
        try:
            plt.imsave(file_path, img if isinstance(img, np.ndarray) else np.array(img))
            logging.info(f"Image {i+1} saved as {file_path}")
        except Exception as e:
            logging.error(f"Failed to save image {i+1}: {str(e)}")

def get_user_selection(images: List[np.ndarray], num_images: int, resolution: int) -> Optional[List[np.ndarray]]:
    """
    Get user's image selection.
    
    Args:
        images (List[np.ndarray]): List of images to choose from.
        num_images (int): Total number of images.
        resolution (int): Resolution of the images.
    
    Returns:
        Optional[List[np.ndarray]]: List of selected images or None if no selection is made.
    """
    while True:
        choice = input(f"Select your favorite images (1-{num_images}), separated by commas, or type 'stop' to exit: ").strip()

        if choice.lower() == 'stop':
            return None

        try:
            selected_indices = [int(x) - 1 for x in choice.split(',')]
            selected_images = [images[i] for i in selected_indices if 0 <= i < num_images]
            
            if not selected_images:
                raise ValueError("No valid images selected")
            
            rename_selected_images(selected_indices, resolution)
            return selected_images
        except ValueError as e:
            logging.warning(f"Invalid input: {e}")
            print(f"Invalid input: {e}. Please enter numbers between 1 and {num_images} separated by commas or 'stop' to exit.")

def rename_selected_images(selected_indices: List[int], resolution: int) -> None:
    """
    Rename selected images.
    
    Args:
        selected_indices (List[int]): Indices of selected images.
        resolution (int): Resolution of the images.
    """
    for i in selected_indices:
        old_path = os.path.join(IMAGE_FOLDER, f"{resolution}-{i+1}.png")
        new_path = os.path.join(IMAGE_FOLDER, f"{resolution}-selected-{i+1}.png")
        try:
            os.rename(old_path, new_path)
            logging.info(f"Image {i+1} renamed to '{os.path.basename(new_path)}'")
        except Exception as e:
            logging.error(f"Failed to rename image {i+1}: {str(e)}")
