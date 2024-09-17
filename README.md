# whisper-transcriber-telegram-bot

A local Whisper AI transcriber bot for Telegram, utilizing GPU or CPU for processing. 

Runs on Python 3.10+.

## About

This program is a Whisper AI-based transcriber Telegram Bot running on Python (v3.10+), designed to transcribe audio from various media sources supported by `yt-dlp`, or via Telegram's audio messages and over audio file uploads (mp3, wav). 

The bot supports a broad range of media sites via `yt-dlp` ([listed here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)), leveraging a locally run OpenAI's Whisper model to process audio and return the transcription in multiple formats.

## Features

- 🎥 Downloads and processes media URLs from any source supported by `yt-dlp`.
- 📲 Can receive Telegram audio messages as well as files in i.e. `.mp3` and `.wav` (+ other `ffmpeg` supported) formats for transcription.
- 🤖 Uses a local Whisper model from the `openai-whisper` package for transcription.
   - _(no API required, use your own PC & available CUDA GPU!)_
- 🖥️ Automatically uses `GPUtil` to map out the best available CUDA-enabled local GPU.
   - _(auto-switching to CPU-only mode if no CUDA GPU is available)_
- 📝 Transcribes audio using OpenAI's Whisper model (can be user-selected with `/model`).
   - _(see [openai/whisper](https://github.com/openai/whisper/) for more info on Whisper)_
- 📄 Returns transcription in text, SRT, and VTT formats.
- 🔄 Handles concurrent transcription requests efficiently with async & task queuing.
- 🕒 Features an asynchronous automatic queue system to manage multiple transcription requests seamlessly.

## Installation (non-Docker version)

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

3. Install [ffmpeg](https://ffmpeg.org) (required for media processing):

   On Ubuntu or Debian tree Linux:
   ```bash
   sudo apt update && sudo apt install ffmpeg
   ```

   On Arch Linux:   
   ```bash
   sudo pacman -S ffmpeg
   ```

   On macOS using Homebrew:
   ```bash
   brew install ffmpeg
   ```

   On Windows using Chocolatey:
   ```bash
   choco install ffmpeg
   ```
   
   On Windows using Scoop:   
   ```bash
   scoop install ffmpeg
   ```

4. Set up your Telegram bot token either in `config/bot_token.txt` or as an environment variable `TELEGRAM_BOT_TOKEN`.

5. Run the bot:
   ```bash
   python src/main.py
   ```

## Dockerized Installation

### Prerequisites

- Docker installed on your machine.
- Docker Compose (optional, for ease of handling environment variables and settings).

### Building the Docker Image

1. Navigate to the root directory of the project where the `Dockerfile` is located.
2. Build the Docker image using the following command:

   ```bash
   docker build -t whisper-transcriber-telegram-bot .
   ```

   This command builds a Docker image named `whisper-transcriber-telegram-bot` based on the instructions in your `Dockerfile`.

### Running the Bot Using Docker

To run the bot using Docker:

```bash
docker run --name whisper-bot -d \
  -e TELEGRAM_BOT_TOKEN='YourTelegramBotToken' \
  -v $(pwd)/config:/app/config \
  -v whisper_cache:/root/.cache/whisper \
  whisper-transcriber-telegram-bot
```

Replace `'YourTelegramBotToken'` with your actual Telegram bot token. This command also mounts the `config` directory and the Whisper model cache directory to preserve settings and downloaded models across container restarts.

## Usage

After launching the bot, you can interact with it via Telegram:

1. Send a video URL, voice message or audio file (in WAV/MP3 format) to the bot.
2. The bot will acknowledge the request and begin processing.
3. Once processing is complete, the bot will send the transcription files to you.

## Commands

- `/info` to view current settings, uptime, GPU info and queue status
- `/help` and `/about` - get help on bot use, list version number, available models and commands, etc.
- `/model` - view the model in usedef process_url or change to another available model.
- `/language` - set the model's transcription language (`auto` =  autodetect); if you know the language spoken in the audio, setting the transcription language manually with this command may improve both transcription speed and accuracy.

## Changes
- v0.1652 - maximum file size checks (20MB) as per to Telegram API
- v0.1651 - improved parsing for Youtube URLs, better error handling
- v0.165 - select allowed audio types if transcription from audio files is enabled
   - (default formats: `mp3, wav, m4a, aac, flac, ogg, wma, aiff`, can be expanded to any `ffmpeg` supported format)
- v0.1603 - error message to the user whenever cookies are needed for `yt-dlp`
- v0.1602 - adjustable transcription completion message (in use y/n, its contents) in `config.ini`
- v0.1601 - process order fixes for transcripts (send as msg <> file)
- v0.16 - added configurable cooldowns & rate limits, see `config.ini`:
   - under `[RateLimitSettings]`: `cooldown_seconds`, `max_requests_per_minute`
- v0.15 - added `config.ini` options `sendasfiles` and `sendasmessages`
   - can be set to `true` or `false` depending on your preferences
   - `sendasmessages` (when set to `true`) sends the transcripts as Telegram messages in chat
   - `sendasfiles` (when set to `true`) sends the transcripts as `.stt`, `.vtt` and `.txt`
   - small fixes to i.e. url handling (`allowallsites` checks; YouTube)
- v0.14.6 - fixed occasional queue hangs with sent audio files (wav/mp3)
- v0.14.5 - fixed following the "keep/don't keep audio files" config rule
- v0.14.4 - added the `/info` command for viewing current settings & queue status
- v0.14.3 - Whisper model language selection via `/language` command
- v0.14.2 - display duration & estimates
- v0.14.1 - small fixes to the file handler; more detailed exception catching
- v0.14 - now handles both Telegram's audio messages as well as audio files (.wav, .mp3)
- v0.13 - added `GPUtil` GPU mapping to figure out the best available CUDA GPU instance to use
   - (by default, uses a CUDA-enabled GPU on the system with the most free VRAM available)
- v0.12 - async handling & user model change fixes, improved error handling
- v0.11.1 - bot logic + layout changes, model list with `/model` (also in `config.ini`)
- v0.11 - bugfixes & rate limits for `/model` command changes for users
- v0.10 - `/help` & `/about` commands added for further assistance
   - `config.ini` now has a list of supported models that can be changed as needed
- v0.09 - users can now change the model Whisper model with `/model` command
- v0.08 - auto-retry TG connection on start-up connection failure
   - can be set in `config.ini` with `RestartOnConnectionFailure`
- v0.07.7 - log output from `whisper` to logging
- v0.07.6 - update interval for logging `yt-dlp` downloads now configurable from `config.ini`
- v0.07.5 - 10-second interval update for `yt-dlp` logging
- v0.07.4 - fixes for non-youtube urls
- v0.07.2 - job queues fine-tuned to be more informative
- v0.07.1 - job queues introduced
- v0.07 - transcript queuing, more precise transcript time estimates
- v0.06 - better handling of details for all video sources, transcription time estimates
- v0.05 - universal video description parsing (platform-agnostic)
- v0.04.1 - version number printouts and added utils
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
- Thanks for additional code contributions: [GRbit](https://github.com/GRbit) (Dockerization)