import whisper

# List all available Whisper models
available_models = whisper.available_models()

print("Available Whisper models:")
for model in available_models:
    print(model)
