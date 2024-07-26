# image_enhancement.py

import logging
from typing import Tuple
import numpy as np
from PIL import Image
import torch
import cv2
from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline, ControlNetModel
from torchvision import transforms
from config import (
    LOG_FORMAT, LOG_DATE_FORMAT, UPSCALER_MODEL, SD_BASE_MODEL,
    CONTROLNET_MODEL, CONTROLNET_CONDITIONING_SCALE,
    CONTROL_GUIDANCE_START, CONTROL_GUIDANCE_END
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

def latent_upscale_image(image: Image.Image, prompt: str, output_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """
    Upscale the given image using Stable Diffusion x2 latent upscaler.
    
    Args:
        image: PIL Image to upscale.
        prompt: Text prompt for guided upscaling.
        output_size: Desired output size as a tuple (width, height).
    
    Returns:
        np.ndarray: Upscaled image as a numpy array.
    """
    try:
        sd_pipeline = StableDiffusionPipeline.from_pretrained(SD_BASE_MODEL, torch_dtype=dtype)
        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(UPSCALER_MODEL, torch_dtype=dtype)
        sd_pipeline.to(device)
        upscaler.to(device)
        
        # Encode the image to latent space
        with torch.no_grad():
            latent = sd_pipeline.vae.encode(transforms.ToTensor()(image).unsqueeze(0).to(device) * 2 - 1)
            latent = latent.latent_dist.sample() * sd_pipeline.vae.config.scaling_factor

        # Upscale the latent
        with torch.no_grad():
            upscaled_latent = upscaler(
                prompt=prompt,
                image=latent,
                num_inference_steps=20,
                guidance_scale=0,
            ).images[0]

        # Decode the upscaled latent
        with torch.no_grad():
            image = sd_pipeline.vae.decode(upscaled_latent / sd_pipeline.vae.config.scaling_factor, return_dict=False)[0]
        image = sd_pipeline.image_processor.postprocess(image, output_type="pil")[0]
        image = image.resize(output_size, Image.LANCZOS)
        
        logger.info(f"Image upscaled successfully to {output_size[0]}x{output_size[1]}")
        return np.array(image)
    except Exception as e:
        logger.error(f"Error during latent image upscaling: {str(e)}")
        logger.info("Falling back to simple resize")
        return np.array(image.resize(output_size, Image.LANCZOS))

def apply_freestyle(image: np.ndarray, prompt: str) -> np.ndarray:
    """
    Apply Freestyle to the given image.
    
    Args:
        image: Input image as a numpy array.
        prompt: Text prompt for Freestyle.
    
    Returns:
        np.ndarray: Processed image as a numpy array.
    """
    # TODO: Implement Freestyle functionality
    logger.info("Freestyle functionality not yet implemented")
    return image

def apply_controlnet(image: np.ndarray, prompt: str) -> np.ndarray:
    """
    Apply ControlNet to the given image.
    
    Args:
        image: Input image as a numpy array.
        prompt: Text prompt for ControlNet.
    
    Returns:
        np.ndarray: Processed image as a numpy array.
    """
    try:
        # Load ControlNet model
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=dtype)
        
        # Load Stable Diffusion pipeline
        pipe = StableDiffusionPipeline.from_pretrained(SD_BASE_MODEL, controlnet=controlnet, torch_dtype=dtype)
        pipe.to(device)
        
        # Prepare control image (Canny edge detection)
        control_image = get_canny_image(image)
        
        # Generate image
        output = pipe(
            prompt,
            image=control_image,
            num_inference_steps=20,
            controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
            control_guidance_start=CONTROL_GUIDANCE_START,
            control_guidance_end=CONTROL_GUIDANCE_END
        ).images[0]
        
        logger.info("ControlNet processing completed successfully")
        return np.array(output)
    except Exception as e:
        logger.error(f"Error during ControlNet processing: {str(e)}")
        return image

def get_canny_image(image: np.ndarray) -> Image.Image:
    """
    Apply Canny edge detection to the input image.
    
    Args:
        image: Input image as a numpy array.
    
    Returns:
        PIL.Image.Image: Canny edge detected image.
    """
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)
