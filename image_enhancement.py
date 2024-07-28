import logging
from typing import Tuple, List
import numpy as np
from PIL import Image
import torch
import cv2
import json
from diffusers import DiffusionPipeline, StableDiffusionLatentUpscalePipeline, StableDiffusionControlNetPipeline, ControlNetModel
from torchvision import transforms
from config import (
    LOG_FORMAT, LOG_DATE_FORMAT, UPSCALER_MODEL, SD_BASE_MODEL,
    CONTROLNET_MODEL, CONTROLNET_CONDITIONING_SCALE,
    CONTROL_GUIDANCE_START, CONTROL_GUIDANCE_END,
    FREESTYLE_PROMPT_JSON,
    FREESTYLE_N, FREESTYLE_B, FREESTYLE_S,
    MODEL_HIGH_RES
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {device_name}")
        if 'AMD' in device_name or 'MI' in device_name:
            logger.info(f"AMD GPU detected. ROCm version: {torch.version.hip}")
            torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        logger.info("GPU not available. Using CPU")
    return device

device = get_device()

# Use mixed precision for GPU, full precision for CPU
dtype = torch.float16 if device.type == "cuda" else torch.float32

def load_style_prompts() -> List[dict]:
    """Load and return the style prompts from the JSON file."""
    try:
        with open(FREESTYLE_PROMPT_JSON, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading style prompts: {str(e)}")
        return []

def select_style_prompt(style_prompts: List[dict]) -> dict:
    """Prompt the user to select a style from the available options."""
    print("Available styles:")
    for i, style in enumerate(style_prompts):
        print(f"{i + 1}. {style['name']}")
    
    while True:
        try:
            choice = int(input("Select a style by number: ")) - 1
            if 0 <= choice < len(style_prompts):
                return style_prompts[choice]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def apply_freestyle(image: np.ndarray, prompt: str) -> np.ndarray:
    """
    Apply Freestyle to the given image using the SDXL model from Hugging Face.
    
    Args:
        image: Input image as a numpy array.
        prompt: Text prompt for Freestyle.
    
    Returns:
        np.ndarray: Processed image as a numpy array.
    """
    try:
        # Load the pipeline directly from Hugging Face
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype)
        pipe.to(device)

        # Load style prompts
        style_prompts = load_style_prompts()
        if not style_prompts:
            raise ValueError("No style prompts available.")

        # Let the user select a style
        selected_style = select_style_prompt(style_prompts)

        # Combine user prompt with selected style prompt
        full_prompt = f"{prompt}, {selected_style['prompt']}"

        # Prepare the input image
        input_image = Image.fromarray(image).resize((1024, 1024))

        # Generate the stylized image
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=dtype):
                    result = pipe(
                        prompt=full_prompt,
                        image=input_image,
                        num_inference_steps=10,
                        guidance_scale=5,
                        num_images_per_prompt=1,
                        negative_prompt=selected_style.get('negative_prompt', ''),
                    ).images[0]
            else:
                result = pipe(
                    prompt=full_prompt,
                    image=input_image,
                    num_inference_steps=10,
                    guidance_scale=5,
                    num_images_per_prompt=1,
                    negative_prompt=selected_style.get('negative_prompt', ''),
                ).images[0]

        logger.info(f"Freestyle processing completed successfully using {selected_style['name']} style")
        return np.array(result)
    except Exception as e:
        logger.error(f"Error during Freestyle processing: {str(e)}")
        return image

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
        sd_pipeline = StableDiffusionLatentUpscalePipeline.from_pretrained(UPSCALER_MODEL, torch_dtype=dtype)
        sd_pipeline.to(device)
        
        # Upscale the image
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=dtype):
                    upscaled_image = sd_pipeline(
                        prompt=prompt,
                        image=image.resize((512, 512)),  # Resize to 512x512 as required by the upscaler
                        num_inference_steps=10,
                        guidance_scale=0,
                    ).images[0]
            else:
                upscaled_image = sd_pipeline(
                    prompt=prompt,
                    image=image.resize((512, 512)),
                    num_inference_steps=10,
                    guidance_scale=0,
                ).images[0]
        
        upscaled_image = upscaled_image.resize(output_size, Image.LANCZOS)
        
        logger.info(f"Image upscaled successfully to {output_size[0]}x{output_size[1]}")
        return np.array(upscaled_image)
    except Exception as e:
        logger.error(f"Error during latent image upscaling: {str(e)}")
        logger.info("Falling back to simple resize")
        return np.array(image.resize(output_size, Image.LANCZOS))

def apply_controlnet(image: np.ndarray, prompt: str) -> np.ndarray:
    """
    Apply ControlNet to the given image and output at 1024x1024 resolution.
    
    Args:
        image: Input image as a numpy array.
        prompt: Text prompt for ControlNet.
    
    Returns:
        np.ndarray: Processed image as a numpy array at 1024x1024 resolution.
    """
    try:
        # Load ControlNet model
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=dtype)
        
        # Load Stable Diffusion pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD_BASE_MODEL,
            controlnet=controlnet,
            torch_dtype=dtype
        )
        pipe.to(device)
        
        # Prepare control image (Canny edge detection)
        control_image = get_canny_image(image)
        
        # Resize control image to 1024x1024
        control_image = control_image.resize((1024, 1024), Image.LANCZOS)
        
        # Generate image
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=dtype):
                    output = pipe(
                        prompt,
                        image=control_image,
                        num_inference_steps=10,
                        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                        control_guidance_start=CONTROL_GUIDANCE_START,
                        control_guidance_end=CONTROL_GUIDANCE_END,
                        height=1024,
                        width=1024
                    ).images[0]
            else:
                output = pipe(
                    prompt,
                    image=control_image,
                    num_inference_steps=10,
                    controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                    control_guidance_start=CONTROL_GUIDANCE_START,
                    control_guidance_end=CONTROL_GUIDANCE_END,
                    height=1024,
                    width=1024
                ).images[0]
        
        logger.info("ControlNet processing completed successfully at 1024x1024 resolution")
        return np.array(output)
    except Exception as e:
        logger.error(f"Error during ControlNet processing: {str(e)}")
        return cv2.resize(image, (1024, 1024))  # Fallback to simple resize

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

def apply_pixart(image: np.ndarray, prompt: str, temperature: float) -> np.ndarray:
    """
    Apply Pixart enhancement to the given image.
    
    Args:
        image: Input image as a numpy array.
        prompt: Text prompt for Pixart.
        temperature: Temperature for generation guidance.
    
    Returns:
        np.ndarray: Processed image as a numpy array.
    """
    try:
        # Load the PixArtAlpha pipeline for high resolution
        pipe = DiffusionPipeline.from_pretrained(MODEL_HIGH_RES, torch_dtype=dtype)
        pipe.to(device)

        # Prepare the input image
        input_image = Image.fromarray(image).resize((1024, 1024))

        # Generate the enhanced image
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=dtype):
                    result = pipe(
                        prompt=prompt,
                        image=input_image,
                        num_inference_steps=10,
                        guidance_scale=temperature,
                        num_images_per_prompt=1,
                    ).images[0]
            else:
                result = pipe(
                    prompt=prompt,
                    image=input_image,
                    num_inference_steps=10,
                    guidance_scale=temperature,
                    num_images_per_prompt=1,
                ).images[0]

        logger.info("Pixart enhancement completed successfully")
        return np.array(result)
    except Exception as e:
        logger.error(f"Error during Pixart enhancement: {str(e)}")
        return image

def apply_enhancement(image: np.ndarray, prompt: str, enhancement_option: str, temperature: float = 5.0) -> np.ndarray:
    """Apply the selected enhancement to the image."""
    if enhancement_option == "Freestyle":
        return apply_freestyle(image, prompt)
    elif enhancement_option == "Upscaler":
        return latent_upscale_image(Image.fromarray(image), prompt)
    elif enhancement_option == "ControlNet":
        return apply_controlnet(image, prompt)
    elif enhancement_option == "Pixart":
        return apply_pixart(image, prompt, temperature)
    else:  # None
        return image

# Print PyTorch and CUDA versions for debugging
logger.info(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")
