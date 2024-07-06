from image_generation_loop import image_generation_loop

def main():
    initial_prompt = "A magical landscape with a towering, crystalline mountain and a serene lake reflecting iridescent clouds. At the lake's edge, a glowing blue-leaved tree stands amidst bioluminescent plants. A bridge of light arches over the lake towards an enchanted castle nestled in the mountain peaks."
    final_images = image_generation_loop(initial_prompt)
    if final_images:
        print("Image generation completed successfully.")
    else:
        print("Program exiting.")

if __name__ == "__main__":
    main()
