from cldm import CLDM, generate_cascaded_images

# Initialize the CLDM model globally to load the models only once
cldm = CLDM("PixArt-alpha/PixArt-XL-2-512x512", "PixArt-alpha/PixArt-XL-2-1024-MS")

def generate_images(prompt, num_images=1, resolution=256, temp=0.7, base_image=None):
    """Generates a specified number of images at a specified resolution."""
    images = generate_cascaded_images(cldm, prompt, [256, 512, 1024], [3, 2, 1], temp)
    return images
