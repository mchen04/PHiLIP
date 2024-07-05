import numpy as np
import torch
from diffusers import PixArtAlphaPipeline
from torchvision import transforms
from PIL import Image

# Preprocessing pipeline for images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained pipelines with debugging
def load_pipeline_with_debugging(model_name):
    model = PixArtAlphaPipeline.from_pretrained(model_name)
    return model

pipe_mid_res = load_pipeline_with_debugging("PixArt-alpha/PixArt-XL-2-512x512")
pipe_high_res = load_pipeline_with_debugging("PixArt-alpha/PixArt-XL-2-1024-MS")

def generate_images(prompt, num_images=1, resolution=512, temp=0.7, base_images=None, steps=50):
    """
    Generates images using the PixArtAlphaPipeline based on the given prompt.
    
    Parameters:
    - prompt: The textual prompt to generate images from.
    - num_images: Number of images to generate.
    - resolution: Resolution of the images (512 or 1024).
    - temp: Temperature for the generation process.
    - base_images: List of base images for conditioning the generation.
    - steps: Number of inference steps.

    Returns:
    - images: List of generated images as numpy arrays.
    """
    # Select the pipeline based on the desired resolution
    if resolution == 512:
        pipe = pipe_mid_res
    elif resolution == 1024:
        pipe = pipe_high_res
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    if base_images is not None:
        base_image_tensors = []
        for base_image in base_images:
            if isinstance(base_image, np.ndarray):
                base_image = Image.fromarray(base_image)
            base_image_tensor = preprocess(base_image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
            base_image_tensors.append(base_image_tensor)
        input_images = torch.mean(torch.stack(base_image_tensors), dim=0)
    else:
        input_images = None

    if torch.cuda.is_available():
        pipe.to("cuda")
    else:
        pipe.to("cpu")

    prompts = [prompt] * num_images

    with torch.no_grad():
        results = pipe(prompts, num_images=num_images, resolution=resolution, guidance_scale=temp, base_image=input_images, return_tensors=False, num_inference_steps=steps)
        images = [np.array(image) for image in results.images]

    return images
