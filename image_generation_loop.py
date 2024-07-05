import os
import shutil
from generate_image import generate_images
from display_image import display_and_select_image
from user_input_handler import handle_user_input
from var_model_processing import process_images, generate_final_image
from var_model_setup import vae, var  # Import the models
import torch
from torchvision import transforms

def clear_generated_images_folder(folder="generated_images"):
    """Clears all files in the generated_images folder."""
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def image_generation_loop():
    while True:
        clear_generated_images_folder()

        prompt = input("Enter a prompt for image generation: ").strip()
        
        while True:
            try:
                temperature = float(input("Enter the temperature for image generation (suggested range from 0.5 to 1.5): ").strip())
                if 0.5 <= temperature <= 1.5:
                    break
                else:
                    print("Please enter a temperature within the suggested range from 0.5 to 1.5.")
            except ValueError:
                print("Invalid input. Please enter a valid number for temperature.")

        selected_images = []
        resolutions = [512, 1024]
        num_images_list = [9, 1]
        inference_steps = [4, 10]
        iteration = 0

        while iteration < len(resolutions):
            resolution = resolutions[iteration]
            num_images = num_images_list[iteration]
            steps = inference_steps[iteration]
            base_images = selected_images if selected_images and iteration > 0 else None

            print(f"Generating images for prompt: {prompt}")
            print(f"Temperature for this iteration: {temperature}")

            generated_images = generate_images(prompt, num_images=num_images, resolution=resolution, temp=temperature, base_images=base_images, steps=steps)

            selected_images = display_and_select_image(generated_images, resolution, iteration)

            if selected_images is None:
                print("No images selected. Restarting with a new prompt and temperature.")
                break

            if iteration == 1:  # After selecting 512 resolution images
                # Process the selected images with the VAR and VAE models
                selected_image_paths = [os.path.join("generated_images", f"{resolution}-selected-{i+1}.png") for i in range(len(selected_images))]
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Extract concepts using VQVAE
                concepts = process_images(selected_image_paths, vae, var, device)

                # Save concept images
                concept_folder = "concept_images"
                if not os.path.exists(concept_folder):
                    os.makedirs(concept_folder)
                    
                for i, concept in enumerate(concepts):
                    concept_image_path = os.path.join(concept_folder, f"concept_{i+1}.png")
                    concept_image = transforms.ToPILImage()(concept.cpu().squeeze(0))
                    concept_image.save(concept_image_path)
                    print(f"Concept image {i+1} saved as {concept_image_path}")

                # Generate final image using the extracted concepts
                final_image = generate_final_image(concepts, prompt, temperature)
                
                # Save the final image
                final_image_path = os.path.join("generated_images", "final_image.png")
                final_image.save(final_image_path)
                print(f"Final image generated and saved as {final_image_path}")

                return final_image

            iteration += 1

        if iteration == len(resolutions):
            print("Image generation completed successfully.")

