from image_generation_loop import image_generation_loop

def main():
    prompt = "From above, Shanghai unfolds where tradition meets the future. Neon dragons twist around skyscrapers, casting an otherworldly glow. Stone lions oversee silk markets under neon's flicker, merging past with cybernetic prospects. The vast Huangpu River mirrors a dance of lights, blending centuries in its reflection, depicting a cityscape where historical and futuristic elements coexist in a vibrant tableau."
    
    final_images = image_generation_loop(prompt)
    if final_images:
        print("Image generation completed successfully.")
    else:
        print("Image generation did not complete successfully.")

if __name__ == "__main__":
    main()
