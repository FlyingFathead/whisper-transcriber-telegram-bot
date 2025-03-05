# whisper-transcriber-telegram-bot

A local Whisper AI transcriber bot for Telegram, utilizing local GPU's or CPU for processing. No Whisper API access required -- just utilize whatever available hardware you have. The bot uses `GPUtil` to automatically select an available CUDA GPU, or reverts to CPU-only if none is found.

**Runs on the latest OpenAI Whisper v3 `turbo` model by default** (as of September 2024). Models can be selected both from `config.ini` and i.e. with the user command `/model`.

The program has been written in Python and it works on Python version `3.10+`, tested up to Python `3.12.2`.

Designed for transcribing audio from various media source, such as URLs supported by `yt-dlp`, or via Telegram's audio messages or over media file uploads on Telegram (`.mp3`, `.wav`, `.ogg`, `.flac`, etc.)

Sites supported by `yt-dlp` are [listed here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md). Supported audio and video file uploads can be configured separately from `config.ini`. Compatible with all `ffmpeg`-supported media formats), with up to `20MB` direct file transfer sizes as supported by Telegram in their bot API.

Can be safely installed and deployed with [Docker](https://www.docker.com/) by using the included `Dockerfile`.

## Features

- 🚀 **(New!)** Now supporting OpenAI's `turbo` model of the Whisper v3 series
   - (Whisper-v3 `turbo` released on September 30, 2024)
   - 8x transcription speed (vs. real-time)
   - Nearly on par with the previous `v3-large` model with only 6GB VRAM usage
- 🎥 Downloads and processes media URLs from any source supported by `yt-dlp`
   - _(can be configured to use `cookies.txt` in `config.ini` for better availability)_
- 📲 Can receive Telegram audio messages as well as files, i.e. `.mp3` and `.wav` for transcription
   - Direct video file uploads in supported media formats is also supported
   - _(all other `ffmpeg` supported formats also configurable via `config.ini`)_
- 🤖 Uses a local Whisper model from the `openai-whisper` package for transcription
   - _(no API required, use your own PC & available CUDA GPU!)_
- 🖥️ Automatically uses `GPUtil` to map out the best available CUDA-enabled local GPU
   - _(auto-switching to CPU-only mode if no CUDA GPU is available)_
- 📝 Transcribes audio using OpenAI's Whisper model (can be user-selected with `/model`)
   - _(see [openai/whisper](https://github.com/openai/whisper/) for more info on Whisper)_
- 📄 Returns transcription in text, SRT, and VTT formats
- 🔄 Handles concurrent transcription requests efficiently with async & task queuing
- 🕒 Features an asynchronous automatic queue system to manage multiple transcription requests seamlessly

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

---
## Dockerized Installation
---

### Prerequisites

- Docker installed on your machine
- If you want to run your Whisper transcripts GPU accelerated with CUDA (recommended), you'll need a Nvidia GPU and the NVIDIA Container Toolkit installed on the host machine that is running the Docker container

---

To enable GPU processing inside Docker files, install the NVIDIA Container Toolkit in i.e. Ubuntu **(on the host machine)** with these steps:

1. **Add NVIDIA GPG Key and Repository**:
   Use the following commands to configure the repository securely with the GPG key:

   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg &&
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   ```

2. **Update the Package List**:
   Run the following to refresh your package list:
   ```bash
   sudo apt-get update
   ```

3. **Install the NVIDIA Container Toolkit**:
   Install the toolkit using:
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   ```

4. **Configure Docker to Use NVIDIA Runtime**:
   Configure the NVIDIA runtime for Docker:
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   ```

5. **Restart Docker**:
   Restart the Docker service to apply the changes:
   ```bash
   sudo systemctl restart docker
   ```

6. **Test the Setup**:
   You can verify if the setup is working correctly by running a base CUDA container:
   ```bash
   sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu22.04 nvidia-smi
   ```

If everything is set up correctly, you should see your GPUs listed.

---
#### Dockerfile Install Option 1: Pull the prebuilt image from GHCR

Just grab the latest pre-built version with:

   ```bash
   docker pull ghcr.io/flyingfathead/whisper-transcriber-telegram-bot:latest
   ```
---
#### Dockerfile Install Option 2: Build the Docker image yourself

If there's something wrong with GHCR's prebuilt image, you can also build the Docker image yourself.

1. Navigate to the root directory of the project where the `Dockerfile` is located.
2. Build the Docker image using the following command:

   ```bash
   docker build -t whisper-transcriber-telegram-bot .
   ```

   This command builds a Docker image named `whisper-transcriber-telegram-bot` based on the instructions in your `Dockerfile`.

---
### Running the Bot Using Docker

To run the bot using Docker (may require `sudo`, depending on whether or not you're using a `docker` group or not):

```bash
docker run --gpus all --name whisper-transcriber-telegram-bot -d \
  -e TELEGRAM_BOT_TOKEN='YourTelegramBotToken' \
  -v whisper_cache:/root/.cache/whisper \
  ghcr.io/flyingfathead/whisper-transcriber-telegram-bot:latest

```

Replace `'YourTelegramBotToken'` with your actual Telegram bot token. This command also mounts the `config` directory and the Whisper model cache directory to preserve settings and downloaded models across container restarts.

 ## Getting the Telegram Bot API Token

1. If you haven't got an active [Telegram Bot API](https://core.telegram.org/bots/api) token yet, set up a new Telegram bot by interacting with Telegram's `@BotFather`. Use Telegram's user lookup to search for the user, message it and run the necessary bot setup to get your API key.

2. After setting up your bot and receiving your Telegram Bot API token from `@BotFather`, either copy-paste the Telegram Bot API authentication token into a text file (`config/bot_token.txt`) or set the API token as an environment variable with the name `TELEGRAM_BOT_TOKEN`. The program will look for both during startup, and you can choose whichever you want.

## Usage

After launching your bot successfully, you can interact with it via Telegram (send a message to `@your_bot_name_Bot`, or whatever your bot name is):

1. Send a video URL (for `yt-dlp` to download), a voice message or an audio file (i.e. `.wav` or `.mp3` format) to the bot.
2. The bot will acknowledge the request and begin processing, notifying the user of the process.
3. Once processing is complete, the bot will send the transcription to you. By default, the transcription is sent as a message as well as `.txt`, `.srt` and `.vtt` files. Transcription delivery formats can be adjusted from the `config.ini`.

## Commands

- `/info` to view current settings, uptime, GPU info and queue status
- `/help` and `/about` - get help on bot use, list version number, available models and commands, etc.
- `/model` - view the model in usedef process_url or change to another available model.
- `/language` - set the model's transcription language (`auto` =  autodetect); if you know the language spoken in the audio, setting the transcription language manually with this command may improve both transcription speed and accuracy.

## Updating
**(New in v0.1712)** You can now update your existing or older `config.ini` with the configuration merger tool that is now located in `./src/utils/configmerger.py`. 

While in the project's main directory, simply type i.e.:

```bash
./src/utils/configmerger.py /path/to/your_old_or_custom_config.ini
```

It will merge your own configuration with the project's current `config.ini` under the `config/` directory.

Or, to try a merge of two files by defining the directories:

```bash
./src/utils/configmerger.py /path/to/main/config.ini /path/to/your_config.ini
```

If you just need to see the options, type:

```bash
./src/utils/configmerger.py
```

## Changes
- v0.1712 - More configuration options for user notifications
   - Added two new booleans in `config.ini` under `[NotificationSettings]`:
     - `send_video_info` (default: `true`): Whether the bot should send video metadata (title, duration, channel info, etc.) to the user in Telegram. If `false`, the info is still logged to console, but not sent to the user.
     - `send_detailed_info` (default: `true`): Whether the bot should send a detailed “transcription process info” message (model name, language, estimated time, etc.) to the user in Telegram. Even if disabled, the console/logger still records the detailed info.
   - Fixed a bug related to referencing an uninitialized `detailed_message` variable when `send_detailed_info = false`.
   - Added `./src/utils/configmerger.py` as a tool for easier updating with your own pre-existing configuration files
   - _(There is also `./src/utils/configmerger_alt.py` that uses a separate type of parsing if you really need to keep your comments)_
- v0.1711 - **🍪 Cookie handling is here!**
   - see `config.ini` for `yt-dlp` options to set up your cookies
   - this will make it easier to enable seamless operation with no interruptions when using some video platforms & services
   - use your own `cookies.txt` file or directly link your favorite browser profile for better, uninterrupted access to any online media sources that need session cookies
- v0.1710 - rewrite for chunking logic when sending as messages & re-encoding tool
   - better step-by-step logging, better error catching, better fitting into TG message limits with fallbacks
   - again; please refer to i.e. [Issue #7](https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/issues/7) (and open up a new issue if necessary) if the problem persists
   
   **(other)**
   - included a helper script in `src/utils/reencode_to_target_size.py` for those who can't fit their media sources within Telegram's Bot API's 20 MB limit. 
   - Please use it to recode your stuff before sending it over to your transcriber bot instance if need be.   
   - Run with i.e.:
   ```bash
   python src/utils/reencode_to_target_size.py /path/to/your_input_file
   ```
- v0.1709.2 - up & running greeting is now more prominent w/ both UTC+local times
- v0.1709.1 - increased split message maximum character safe zone buffers to prevent chunk exceeding
   - added a further safeguard to fall back on character-level splitting if no whitespace is found
   - please refer to i.e. [Issue #7](https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/issues/7) (and open up a new issue if necessary) if the problem persists
- v0.1709 - Added `config.ini` option to ping users (i.e. the owner) on startup (when the service is online)
   - startup notifications true/false, user ID's and the environment variable and fallbacks can be defined in `config.ini`
- v0.1708.4 - Better error catching
   - Fixed the description and catching of i.e. YouTube's 403 errors
- v0.1708.3 - Enforced chunk size double-check when sending transcripts as messages
   - This is to ensure we're staying under the message length cap in Telegram
- v0.1708.2 - Added buffer for chunking
   - Changed the chunk sizes from `4096` to `4000` to avoid edge cases
- v0.1708.1 - Small bug fixes in the output
   - Note that running the program within `firejail` using Nvidia driver v.560.xx or newer requires i.e.:
   ```bash
   firejail --noblacklist=/sys/module --whitelist=/sys/module/nvidia* --read-only=/sys/module/nvidia* python src/main.py
   ```
   This is due to recent changes in Nvidia's driver handling on Linux, see i.e. [here](https://github.com/netblue30/firejail/issues/6509) or [here](https://github.com/netblue30/firejail/issues/6372)
   - Dockerized versions should run without problems
- v0.1708 - Direct video file uploads are now available
   - (to prevent abuse, they're disabled by default, see `config.ini`)
- v0.1707 - New `config.ini` option: add sites that require full video download
   - some media sites don't work well with `yt-dlp`'s audio-only download method
   - there are now two new options in `config.ini` under `[YTDLPSettings]`:
   - `download_original_video_for_domains_active = true` (default)
   - `download_original_video_domains = site1.com, site2.com, site3.com`
   - at the moment it's used for media platforms that have had reported issues during testing
   - when active, a comma-separated list is used to check up on media sites that require their contents to be downloaded as the original video instead of audio-only
   - _(the tradeoff is obviously download size and hence speed; the audio-only method is usually the fastest and should be preferred for most popular sites, hence only add problematic sites to the video-only list)_
   - using worst available video quality (with audio in it) is usually recommended
   - video quality selection is in `config.ini`: `use_worst_video_quality = true` (default is true, set to false if it doesn't work for your setup)
   - again, the default setup in this version should work for most users
- v0.1706 - Disable asking for token if running inside Docker
   - by default, the app will ask for the token if it's not found, unless Dockerized
   - can be better for headless use case scenarios where you need the error message rather than a prompt for the bot token
   - `Dockerfile` now has `RUNNING_IN_DOCKER` environment variable set for detection
- v0.1705 - Dockerized pre-builds; thanks to [jonmjr](https://github.com/jonmrjr) for assistance!
   - updated `src/utils/bot_token.py` to query for a bot token if it's not found from either env vars or from the file
   - can be useful when running the bot in Docker containers
   - this option can be set on/off in `config.ini` with `AskForTokenIfNotFound = True` (default is `true`)
- v0.1704 - Token logic / `bot_token.py` updates; added `config.ini` preferences for reading the bot token
   - `preferenvforbottoken = True` is now on by default to prefer the environment variable entry for the bot token.
   - set to `false` to prefer `config/bot_token.txt` over the environment variable
   - `AllowBotTokenFallback = True` to allow fallbacks (whether from the env var to `bot_token.txt` or the other way around)
   - set to `false` to strictly disallow fallback bot API token checking
   - improved error catching + exit logic when the token is not found
- v0.1703 - included and updated welcome message (`/start`)
- v0.17021 - updated model info in `/model`
- v0.1702 - prevent queue hang cases with new method
- v0.1701 - better exception catching when chunking long transcripts (due to Telegram's message limits) [See issue](https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/issues/3)
- v0.17 - (1. Oct 2024) **Now supports OpenAI's brand new Whisper v3 [`turbo` model](https://github.com/openai/whisper/blob/main/model-card.md)**
   - `turbo` is enabled by default
- v0.1658 - `UpdateSettings` setting added to `config.ini` to update your bot on startup (can be set to `True` or `False`), as i.e. `yt-dlp` is highly recommended to be kept up to date constantly. You can modify the command line string to whatever modules you want to check updates on during startup.
   - fixed a parsing bug in YouTube urls
   - bot now announces successful downloads
   - added a few emojis here and there for clarity
      _(feel free to comment if you don't like them)_
- v0.1657 - more verbose error messages from `yt-dlp` if the download failed
- v0.1656 - introduced `safe_split_message` to split transcription better and more reliably (edge cases etc) into chunks when longer transcripts are sent as messages
- v0.1655 - added `diarization.py` and `resemblyzer_safety_check.py` under `src/utils/` for [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) diarization support
   - these are WIP; for future in-bot diarization implementations (requires `pip install resemblyzer` to be installed first in order to run)
   - the current resemblyzer pip version (`resemblyzer==0.1.4`) can be patched with `resemblyzer_safety_check.py` to ensure safe pickle/depickle as per up-to-date standards
   - `diarization.py` can be used as a standalone diarization module for testing (requires `resemblyzer`)
      - (try with i.e. `python diarization.py inputfile.mp3 textfile.txt`)
   - both will pave the way for future diarization options that will be implemented in the bot's functionalities in the future
- v0.1654 - `yt-dlp` can now be configured to use cookies (for i.e. YouTube downloads) in `config.ini`
- v0.1653 - even more exception and error catching, especially for YouTube URLs
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
- v0.04 - expanded support for various media sources via `yt-dlp`, supported sites listed [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)
- v0.03 - better logging to console, Whisper model + keep audio y/n can now be set in `config.ini`
- v0.02 - add video information to the transcript text file 
    - (see: `config.ini` => `IncludeHeaderInTranscription = True`)
- v0.01 - initial commit

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please open an issue or submit a pull request.

## License

Licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## Credits

- [FlyingFathead](https://github.com/FlyingFathead) - Project creator
- ChaosWhisperer - Contributions to the Whisper integration and documentation
- Thanks for additional code contributions: [GRbit](https://github.com/GRbit) (Dockerization), [jonmjr](https://github.com/jonmrjr) (more Dockerization)