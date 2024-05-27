from image_generation_loop import image_generation_loop

def main():
    prompt = ("Above the grey fog lies a serene, otherworldly expanse. The indigo sky glimmers with distant stars and glowing celestial bodies, casting soft hues across floating islands of bioluminescent flora. The silence is profound, broken only by whispers of cosmic winds. This ethereal realm, where reality blurs and ancient magic pervades, unfolds the mysteries of the cosmos in a dance of light and shadow.")

    final_images = image_generation_loop(prompt)
    if final_images:
        print("Image generation completed successfully.")
    else:
        print("Image generation did not complete successfully.")

if __name__ == "__main__":
    main()
