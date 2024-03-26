# whisper-transcriber-telegram-bot

A Python-based Whisper transcriber bot for Telegram.

## About

This is a Whisper AI Transcriber Telegram Bot is a Python-based bot designed to transcribe audio from YouTube videos using OpenAI's Whisper model. Users can send YouTube URLs to the bot, which then processes the audio and returns the transcription in various formats.

## Features

- Processes YouTube URLs sent by users.
- Downloads audio from YouTube videos using `yt-dlp`.
- Uses a local model from the `openai-whisper` package.
- Transcribes audio using the OpenAI's Whisper's local model.
- Returns transcription in text, SRT, and VTT formats.
- Handles concurrent transcription requests gracefully.

## Installation

To set up the Whisper Transcriber Telegram Bot, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/FlyingFathead/whisper-transcriber-telegram-bot.git
   cd whisper-transcriber-telegram-bot
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Telegram bot token either in `config/bot_token.txt` or as an environment variable `TELEGRAM_BOT_TOKEN`.

4. Run the bot:
   ```bash
   python src/main.py
   ```

## Usage

After launching the bot, you can interact with it via Telegram:

1. Send a YouTube video URL to the bot.
2. The bot will acknowledge the request and begin processing.
3. Once processing is complete, the bot will send the transcription files to you.

## Changes

v0.02 - add video information to the transcript text file (`config.ini` => `IncludeHeaderInTranscription = True`)
v0.01 - initial commit

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please open an issue or submit a pull request.

## Credits

- [FlyingFathead](https://github.com/FlyingFathead) - Project creator
- ChaosWhisperer - Contributions to the Whisper integration and documentation
