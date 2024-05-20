import numpy as np
import torch
from diffusers import PixArtAlphaPipeline
from torchvision import transforms
from losses import PerceptualLoss  # Import PerceptualLoss from losses.py
from PIL import Image

# Define transformations for input and target images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize the pipeline globally to load the model only once
pipe_low_res = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512")
pipe_mid_res = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512")
pipe_high_res = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS")

# Define Perceptual Loss
perceptual_loss = PerceptualLoss().to("cuda" if torch.cuda.is_available() else "cpu")

def mask_latent_representation(latent, mask_fraction=0.5):
    """Apply a mask to the latent representation to block part of the information."""
    mask = np.random.rand(*latent.shape) > mask_fraction
    return latent * mask

def generate_images(prompt, num_images=1, resolution=256, temp=0.7, base_image=None, mask_fraction=0.5):
    """Generates a specified number of images at a specified resolution with cascading."""
    if resolution == 256:
        pipe = pipe_low_res
    elif resolution == 512:
        pipe = pipe_mid_res
    elif resolution == 1024:
        pipe = pipe_high_res
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    if base_image is not None:
        if isinstance(base_image, np.ndarray):
            base_image = Image.fromarray(base_image)
        base_image_tensor = preprocess(base_image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        input_image = mask_latent_representation(base_image_tensor, mask_fraction)  # Apply masking to base image
    else:
        input_image = None

    # Enable mixed precision if supported
    if torch.cuda.is_available():
        pipe.to("cuda")
    else:
        pipe.to("cpu")

    # Prepare prompt in a batch
    prompts = [prompt] * num_images

    # Generate images in a single forward pass
    with torch.no_grad():
        results = pipe(prompts, num_images=num_images, resolution=resolution, guidance_scale=temp, base_image=input_image, return_tensors=False)
        images = [np.array(image) for image in results.images]

    # Convert numpy arrays to PIL images before preprocessing
    pil_images = [Image.fromarray(image) for image in images]

    # Convert generated images to tensors
    generated_images_tensor = torch.stack([preprocess(image) for image in pil_images]).to("cuda" if torch.cuda.is_available() else "cpu")

    # Calculate perceptual loss and refine images
    if base_image is not None:
        loss = perceptual_loss(generated_images_tensor, base_image_tensor)
        # Optionally, refine generated images using the loss (this can be a more complex process)
        # For simplicity, we're just printing the loss here
        print("Perceptual Loss:", loss.item())

    return images
