# PHiLIP

An image-generation and enhancement prototype built with Python, PyTorch, and Hugging Face Diffusers.

## Repository contents

- [generate_image.py](generate_image.py) loads PixArt pipelines and generates image arrays from prompts.
- [image_generation_loop.py](image_generation_loop.py) coordinates image selection, enhancement, and saving.
- [image_enhancement.py](image_enhancement.py) contains upscaling, ControlNet, and style-prompt processing.
- [main.py](main.py) declares Flask routes for generation, enhancement, and settings.

## Current limits

The generation route passes multiple arguments to a loop function that accepts one argument.
The dependency snapshot is named [requiremets.txt](requiremets.txt) and includes machine-specific local paths.
Model loading and end-to-end operation are not verified here, so this document provides no installation recipe.
