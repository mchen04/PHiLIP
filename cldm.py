import torch
from torch import nn
from diffusers import PixArtAlphaPipeline
import numpy as np
from PIL import Image

class CLDM(nn.Module):
    def __init__(self, low_res_model_path, high_res_model_path):
        super(CLDM, self).__init__()
        self.low_res_model = PixArtAlphaPipeline.from_pretrained(low_res_model_path)
        self.high_res_model = PixArtAlphaPipeline.from_pretrained(high_res_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, prompt, num_inference_steps=50, guidance_scale=0.7):
        low_res_output = self.low_res_model(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        low_res_images = low_res_output.images

        high_res_images = []
        for low_res_image in low_res_images:
            low_res_image = self.preprocess_image(low_res_image)
            low_res_latent = self.high_res_model.vae.encode(low_res_image).latent_dist.sample()
            high_res_output = self.high_res_model(prompt, latents=low_res_latent, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
            high_res_image = self.postprocess_image(high_res_output.images[0])
            high_res_images.append(high_res_image)

        return high_res_images

    def preprocess_image(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image).transpose(2, 0, 1) / 255.0
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        return image

    def postprocess_image(self, image):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        return image

def generate_cascaded_images(cldm, prompt, resolutions, num_images_list, temp=0.7):
    images = []
    base_image = None

    for resolution, num_images in zip(resolutions, num_images_list):
        if base_image is not None:
            input_image = np.array(base_image)
        else:
            input_image = None

        # Generate low-resolution images
        low_res_output = cldm.low_res_model(prompt, num_inference_steps=50, guidance_scale=temp)
        low_res_images = low_res_output.images

        # Select the first low-resolution image and convert it to latent space
        low_res_image = cldm.preprocess_image(low_res_images[0])
        low_res_latent = cldm.high_res_model.vae.encode(low_res_image).latent_dist.sample()

        # Generate high-resolution images using the latent
        high_res_images = []
        for _ in range(num_images):
            high_res_output = cldm.high_res_model(prompt, latents=low_res_latent, num_inference_steps=50, guidance_scale=temp)
            high_res_image = cldm.postprocess_image(high_res_output.images[0])
            high_res_images.append(high_res_image)

        base_image = high_res_images[0]  # Assuming the first image for simplicity
        images.append(base_image)

    return images
