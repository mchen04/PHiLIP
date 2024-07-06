import logging
from image_generation_loop import image_generation_loop
from config import INITIAL_PROMPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    final_images = image_generation_loop(INITIAL_PROMPT)
    if final_images:
        logging.info("Image generation completed successfully.")
    else:
        logging.info("Program exiting.")

if __name__ == "__main__":
    main()
