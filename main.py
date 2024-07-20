# main.py

import logging
from image_generation_loop import image_generation_loop
from config import INITIAL_PROMPT, LOG_FORMAT, LOG_DATE_FORMAT
from PIL import Image

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

def main() -> None:
    """Main function to run the image generation process."""
    final_image = image_generation_loop(INITIAL_PROMPT)
    if final_image is not None:
        logger.info("Image generation and upscaling completed successfully.")
        Image.fromarray(final_image).save("final_upscaled_image.png")
        logger.info("Final upscaled image saved as 'final_upscaled_image.png'")
    else:
        logger.info("Program exiting.")

if __name__ == "__main__":
    main()
