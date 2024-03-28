# whisper-transcriber-telegram-bot

A Python-based Whisper transcriber bot for Telegram.

## About

This is a Whisper AI-based transcriber Telegram Bot running on Python, designed to transcribe audio from various media sources supported by `yt-dlp`. While initially focused on YouTube, the bot now supports a broad range of sites listed [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md), leveraging OpenAI's Whisper model to process audio and return the transcription in multiple formats.


## Features

- Processes media URLs from a variety of sources supported by `yt-dlp`.
- Downloads audio using `yt-dlp` from supported sites including but not limited to YouTube.
- Uses a local model from the `openai-whisper` package for transcription.
- Transcribes audio using OpenAI's Whisper model.
- Returns transcription in text, SRT, and VTT formats.
- Handles concurrent transcription requests efficiently.

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

- v0.04 - expanded support for various media sources via `yt-dlp`, supported sites listed [here](https://github.com/yt-dlp/yt-dlp/blob/)
- v0.03 - better logging to console, Whisper model + keep audio y/n can now be set in `config.ini`
- v0.02 - add video information to the transcript text file 
    - (see: `config.ini` => `IncludeHeaderInTranscription = True`)
- v0.01 - initial commit

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please open an issue or submit a pull request.

## Credits

- [FlyingFathead](https://github.com/FlyingFathead) - Project creator
- ChaosWhisperer - Contributions to the Whisper integration and documentation
