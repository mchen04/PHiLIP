import logging
from config import VALID_USER_INPUTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def handle_user_input() -> str:
    """
    Handle user input for image generation process.
    
    Returns:
        str: Valid user input option.
    """
    while True:
        user_input = input(
            "Options: 'regenerate', 'reselect', 'stop', 'continue', 'prompt', 'temperature', or 'restart': "
        ).strip().lower()
        if user_input in VALID_USER_INPUTS:
            return user_input
        else:
            logging.warning(f"Invalid option: {user_input}. Please enter a valid command.")
