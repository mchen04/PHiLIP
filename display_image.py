# display_image.py

import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from typing import List, Optional
from PIL import Image
from config import IMAGE_FOLDER, LOG_FORMAT, LOG_DATE_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

def display_and_select_image(images: List[np.ndarray], resolution: int, iteration: int) -> Optional[Image.Image]:
    """
    Displays images, prompts the user to select their favorite, and saves all images.
    
    Args:
        images: List of images to display and save.
        resolution: Resolution of the images.
        iteration: Current iteration number.
    
    Returns:
        Optional[Image.Image]: Selected image as a PIL Image or None if no selection is made.
    """
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    axs = axs if num_images > 1 else [axs]

    for i, img in enumerate(images):
        axs[i].imshow(img)
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
        images: List of images to save.
        resolution: Resolution of the images.
    """
    logger.info(f"Saving {len(images)} images at {resolution}x{resolution} resolution")
    for i, img in enumerate(images):
        file_path = os.path.join(IMAGE_FOLDER, f"{resolution}-{i+1}.png")
        try:
            plt.imsave(file_path, img)
        except Exception as e:
            logger.error(f"Failed to save image {i+1}: {str(e)}")
    logger.info(f"All images saved successfully in {IMAGE_FOLDER}")

def get_user_selection(images: List[np.ndarray], num_images: int, resolution: int) -> Optional[Image.Image]:
    """
    Get user's image selection.
    
    Args:
        images: List of images to choose from.
        num_images: Total number of images.
        resolution: Resolution of the images.
    
    Returns:
        Optional[Image.Image]: Selected image as a PIL Image or None if no selection is made.
    """
    while True:
        choice = input(f"Select your favorite image (1-{num_images}), or type 'stop' to exit: ").strip()

        if choice.lower() == 'stop':
            return None

        try:
            index = int(choice) - 1
            if 0 <= index < num_images:
                selected_image = Image.fromarray(images[index])
                rename_selected_image(index, resolution)
                return selected_image
            else:
                raise ValueError()
        except ValueError:
            logger.warning(f"Invalid input. Please enter a number between 1 and {num_images} or 'stop' to exit.")

def rename_selected_image(index: int, resolution: int) -> None:
    """
    Rename selected image.
    
    Args:
        index: Index of selected image.
        resolution: Resolution of the image.
    """
    old_path = os.path.join(IMAGE_FOLDER, f"{resolution}-{index+1}.png")
    new_path = os.path.join(IMAGE_FOLDER, f"{resolution}-selected.png")
    try:
        os.rename(old_path, new_path)
        logger.info(f"Image {index+1} renamed to '{os.path.basename(new_path)}'")
    except Exception as e:
        logger.error(f"Failed to rename image {index+1}: {str(e)}")
