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
    if resolution == 512:
        pipe = pipe_mid_res
    elif resolution == 1024:
        pipe = pipe_high_res
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    if base_images is not None:
        if isinstance(base_images, list):
            # Process and average multiple base images
            base_image_tensors = []
            for img in base_images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                img_tensor = preprocess(img).unsqueeze(0)
                base_image_tensors.append(img_tensor)
            
            # Stack and average the tensors
            stacked_tensors = torch.stack(base_image_tensors)
            averaged_tensor = torch.mean(stacked_tensors, dim=0)
            
            input_images = averaged_tensor.to("hip" if torch.cuda.is_available() else "cpu")
        else:
            # Single base image
            if isinstance(base_images, np.ndarray):
                base_images = Image.fromarray(base_images)
            input_images = preprocess(base_images).unsqueeze(0).to("hip" if torch.cuda.is_available() else "cpu")
    else:
        input_images = None

    if torch.cuda.is_available():
        pipe.to("hip")
    else:
        pipe.to("cpu")

    prompts = [prompt] * num_images

    with torch.no_grad():
        results = pipe(prompts, num_images=num_images, resolution=resolution, guidance_scale=temp, image=input_images, return_tensors=False, num_inference_steps=steps)
        images = [np.array(image) for image in results.images]

    return images
