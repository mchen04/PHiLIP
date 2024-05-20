def handle_user_input():
    while True:
        user_input = input("Options: type 'regenerate' to recreate images, 'restart' to start over, 'reselect' to choose previous image again, 'stop' to exit, 'prompt' to change prompt, 'temperature' to change temperature, or 'continue' to proceed to the next iteration: ").strip().lower()
        if user_input in {"regenerate", "restart", "reselect", "stop", "prompt", "temperature", "continue"}:
            return user_input
        else:
            print("Invalid option. Please enter a valid command.")
