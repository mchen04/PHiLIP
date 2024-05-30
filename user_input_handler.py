def handle_user_input():
    while True:
        user_input = input("Options: type 'regenerate' to recreate images, 'reselect' to choose previous image again, 'stop' to exit, or 'continue' to proceed to the next iteration: ").strip().lower()
        if user_input in {"regenerate", "reselect", "stop", "continue"}:
            return user_input
        else: 
            print("Invalid option. Please enter a valid command.")
