from diffusers import PixArtAlphaPipeline
import numpy as np

# Initialize the pipeline globally to load the model only once
pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS")

def generate_images(prompt, num_images=1, resolution=256, temp=1.0, base_image=None):
    """Generates a specified number of images at a specified resolution."""
    if base_image is not None:
        input_image = np.array(base_image)  # Convert PIL image to numpy if not already
    else:
        input_image = None

    images = [pipe(prompt, num_images=1, resolution=resolution, guidance_scale=temp, base_image=input_image, return_tensors=False).images[0] for _ in range(num_images)]
    return images
