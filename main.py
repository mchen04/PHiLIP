from generate_image import generate_images
from display_image import display_and_select_image
import torch

def main():
    prompt = "Envision Gloomvale, a city bathed in eternal twilight, dominated by gothic spires and Victorian architecture. Ancient cobblestone streets weave through a dense, omnipresent fog. The towering Clock Tower, constructed from iron and brass, is lit by flickering gas lamps. Ivy-draped buildings cast long shadows, imbuing the air with an aura of mystery."

    selected_image = None
    resolutions = [256, 512, 1024]
    num_images_list = [3, 2, 1]

    for resolution, num_images in zip(resolutions, num_images_list):
        generated_images = generate_images(prompt, num_images=num_images, resolution=resolution, temp=1.0, base_image=selected_image)
        selected_image = display_and_select_image(generated_images, resolution)
        if selected_image is None:
            print("No image selected, stopping.")
            break

if __name__ == "__main__":
    main()