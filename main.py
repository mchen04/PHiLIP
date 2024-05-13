from image_generation_loop import image_generation_loop

def main():
    prompt = 
"Envision a cyberpunk Shanghai at night, where neon blues and purples light up rain-slick skyscrapers and misty streets. The city glows with cybernetic ads and dynamic billboards, illustrating a blend of high technology and urban decay."
    
    final_images = image_generation_loop(prompt)
    if final_images:
        print("Image generation completed successfully.")
    else:
        print("Image generation did not complete successfully.")

if __name__ == "__main__":
    main()
