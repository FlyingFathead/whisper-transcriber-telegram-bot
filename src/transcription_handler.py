# transcription_handler.py
# ~~~
# openai-whisper transcriber-bot for Telegram
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import GPUtil 
import sys
import time
import logging
import re
import threading
import asyncio
from asyncio.exceptions import TimeoutError
import json
import os
import textwrap
import configparser
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta
# import wave
from pydub import AudioSegment

# tg modules // button selection
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# internal modules
from utils.language_selection import ask_language

# config
from config_loader import ConfigLoader
config = ConfigLoader.get_config()

# Toggle this to use the full description or a snippet.
USE_SNIPPET_FOR_DESCRIPTION = config.getboolean('VideoDescriptionSettings', 'use_snippet_for_description', fallback=False)
# If we're using a snippet of the description, maximum number of lines to include
DESCRIPTION_MAX_LINES = config.getint('VideoDescriptionSettings', 'description_max_lines', fallback=30)

# Output directory for transcriptions; create if doesn't exist
output_dir = "transcriptions"
os.makedirs(output_dir, exist_ok=True)

# Define and create audio directory
audio_dir = "audio"
os.makedirs(audio_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# asyncio debugging on
asyncio.get_event_loop().set_debug(True)

# set the config base dir just once at the top of your script
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define a dictionary at the module level to store user-specific models
user_models = {}

# lock the user models
user_models_lock = threading.Lock()

# Define a dictionary at the module level to store user-specific languages
user_languages = {}

# lock the user languages
user_languages_lock = threading.Lock()

# Modify the set_user_language function to use the lock
def set_user_language(user_id, language):
    with user_languages_lock:  # Acquire the lock before modifying user_languages
        global user_languages
        if user_id and language:
            user_languages[user_id] = language
            logger.info(f"Language set for user {user_id}: {language}")
        else:
            logger.error(f"Failed to set language for user {user_id}: {language}")

# get whisper language
def get_whisper_language(user_id=None):
    with user_languages_lock:  # Acquire the lock before accessing user_languages
        logger.debug(f"Attempting to fetch language for user_id: {user_id}")
        global user_languages
        if user_id is None or user_id not in user_languages:
            config = configparser.ConfigParser()
            config_path = os.path.join(base_dir, 'config', 'config.ini')
            config.read(config_path)
            default_language = config.get('WhisperSettings', 'defaultlanguage', fallback='auto')
            logger.info(f"No custom language for user {user_id}. Using default language: {default_language}")
            return default_language
        else:
            custom_language = user_languages[user_id]
            logger.info(f"Returning custom language for user {user_id}: {custom_language}")
            return custom_language

# get the general settings
def get_general_settings():
    config = configparser.ConfigParser()    
    config_path = os.path.join(base_dir, 'config', 'config.ini')
    config.read(config_path)
    allow_all_sites = config.getboolean('GeneralSettings', 'AllowAllSites', fallback=False)
    return {
        'allow_all_sites': allow_all_sites
    }

# get the logging settings
def get_logging_settings():
    config = configparser.ConfigParser()
    config_path = os.path.join(base_dir, 'config', 'config.ini')
    config.read(config_path)
    update_interval = config.getint('LoggingSettings', 'UpdateIntervalSeconds', fallback=10)
    return update_interval

# get whisper model
def get_whisper_model(user_id=None):
    with user_models_lock:  # Acquire the lock before accessing user_models
        logger.debug(f"Attempting to fetch model for user_id: {user_id}")
        global user_models
        if user_id is None or user_id not in user_models:
            config = configparser.ConfigParser()
            config_path = os.path.join(base_dir, 'config', 'config.ini')
            config.read(config_path)
            default_model = config.get('WhisperSettings', 'Model', fallback='medium.en')
            logger.info(f"No custom model for user {user_id}. Using default model: {default_model}")
            return default_model
        else:
            custom_model = user_models[user_id]
            logger.info(f"Returning custom model for user {user_id}: {custom_model}")
            return custom_model

# Modify the set_user_model function to use the lock
def set_user_model(user_id, model):
    with user_models_lock:  # Acquire the lock before modifying user_models
        global user_models
        if user_id and model:
            user_models[user_id] = model
            logger.info(f"Model set for user {user_id}: {model}")
        else:
            logger.error(f"Failed to set model for user {user_id}: {model}")

# get audio duration
def get_audio_duration(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        duration = len(audio) / 1000.0  # duration in seconds
        return duration
    except Exception as e:
        logger.error(f"Error reading audio file {file_path}: {e}")
        return None

# (new) Function to get transcription settings
def get_transcription_settings():
    try:
        config = ConfigLoader.get_config()
        transcription_settings = {
            'include_header': config.getboolean('TranscriptionSettings', 'includeheaderintranscription', fallback=False),
            'keep_audio_files': config.getboolean('TranscriptionSettings', 'keepaudiofiles', fallback=False),
            'send_as_files': config.getboolean('TranscriptionSettings', 'sendasfiles', fallback=True),
            'send_as_messages': config.getboolean('TranscriptionSettings', 'sendasmessages', fallback=False),
        }
        logger.info(f"Loaded transcription settings: {transcription_settings}")
        return transcription_settings
    except Exception as e:
        logger.error(f"Error loading transcription settings: {e}")
        return {
            'include_header': False,
            'keep_audio_files': False,
            'send_as_files': True,
            'send_as_messages': False,
        }

# split long messages
def split_message(message, max_length=3500):
    return [message[i:i+max_length] for i in range(0, len(message), max_length)]

# // audio download (new method)
async def download_audio(url, audio_path):
    config = ConfigLoader.get_config()
    ytdlp_settings = ConfigLoader.get_ytdlp_domain_settings()
    use_cookies = config.getboolean('YTDLPSettings', 'use_cookies', fallback=False)
    cookies_file = config.get('YTDLPSettings', 'cookies_file', fallback='config/cookies.txt')
    use_worst_video_quality = config.getboolean('YTDLPSettings', 'use_worst_video_quality', fallback=True)
    
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    if domain.startswith('www.'):
        domain = domain[4:]  # Remove 'www.'

    should_download_video = ytdlp_settings['active'] and domain in ytdlp_settings['domains']

    if should_download_video:
        logger.info("Identified domain requiring full video download.")
        # Step 1: Get available formats in JSON
        command = [
            "yt-dlp",
            "--no-warnings",
            "--dump-json",
            url
        ]
        if use_cookies and os.path.exists(cookies_file):
            command.extend(["--cookies", cookies_file])

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout_data, stderr_data = await process.communicate()
        if process.returncode != 0:
            stderr_output = stderr_data.decode()
            logger.error(f"Failed to get video formats: {stderr_output}")
            raise Exception(f"Failed to get video formats: {stderr_output}")

        # Step 2: Parse JSON to find the appropriate format
        video_info = json.loads(stdout_data.decode())
        formats = video_info.get('formats', [])

        if not formats:
            raise Exception("No formats found for the video.")

        # Filter out formats without audio
        video_formats = [
            fmt for fmt in formats
            if fmt.get('vcodec') != 'none' and fmt.get('acodec') != 'none' and fmt.get('acodec') != 'video only'
        ]

        if not video_formats:
            raise Exception("No suitable video formats with audio available.")

        if use_worst_video_quality:
            # Sort video formats by resolution (width x height) or bitrate
            def get_format_sort_key(fmt):
                width = fmt.get('width') or 0
                height = fmt.get('height') or 0
                total_pixels = width * height
                tbr = fmt.get('tbr') or 0
                return (total_pixels, tbr)

            selected_format = min(video_formats, key=get_format_sort_key)
            logger.info("Selected worst quality video format.")
        else:
            # Select best quality video format
            def get_format_sort_key(fmt):
                width = fmt.get('width') or 0
                height = fmt.get('height') or 0
                total_pixels = width * height
                tbr = fmt.get('tbr') or 0
                return (-total_pixels, -tbr)

            selected_format = max(video_formats, key=get_format_sort_key)
            logger.info("Selected best quality video format.")

        selected_format_id = selected_format.get('format_id')

        if not selected_format_id:
            raise Exception("Could not determine selected format ID.")

        logger.info(f"Selected format ID: {selected_format_id}")

        # Step 3: Download video using the selected format
        base_output_path = audio_path.replace('.mp3', '')  # e.g., audio/12345_1618033988
        video_output_template = f"{base_output_path}.%(ext)s"  # e.g., audio/12345_1618033988.mp4

        command = [
            "yt-dlp",
            # "--verbose", # uncomment to set verbose
            "--format", selected_format_id,
            "--output", video_output_template,
            url
        ]
        if use_cookies and os.path.exists(cookies_file):
            command.extend(["--cookies", cookies_file])

        logger.info("Downloading the selected quality video with audio...")
    else:
        # Download audio-only as mp3
        command = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--output", audio_path,
            url
        ]
        if use_cookies and os.path.exists(cookies_file):
            command.extend(["--cookies", cookies_file])
        logger.info("Downloading audio-only...")

    # Start the subprocess
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Read and log output
    stdout_lines = []
    stderr_lines = []

    async def read_stream(stream, lines, log_func):
        while True:
            line = await stream.readline()
            if line:
                decoded_line = line.decode().rstrip()
                lines.append(decoded_line)
                log_func(decoded_line)
            else:
                break

    await asyncio.gather(
        read_stream(process.stdout, stdout_lines, logger.info),
        read_stream(process.stderr, stderr_lines, logger.error)
    )

    await process.wait()

    if process.returncode != 0:
        stderr_output = '\n'.join(stderr_lines)
        logger.error(f"yt-dlp failed with error:\n{stderr_output}")
        raise Exception(f"Failed to download media: {stderr_output}")

    if should_download_video:
        # Step 4: Extract audio from the downloaded video
        video_extensions = ['mp4', 'webm', 'mkv', 'avi', 'mov', 'flv', 'wmv', 'mpg', 'mpeg']
        video_file = None
        for ext in video_extensions:
            potential_video = f"{base_output_path}.{ext}"
            if os.path.exists(potential_video):
                video_file = potential_video
                break

        if not video_file:
            logger.error("Failed to locate the downloaded video file.")
            raise Exception("Failed to locate the downloaded video file.")

        logger.info(f"Video file downloaded: {video_file}")

        try:
            logger.info("Starting audio extraction from video file...")
            # Use ffmpeg via pydub to extract audio
            audio = AudioSegment.from_file(video_file)
            logger.info("Audio file loaded, exporting to mp3...")
            audio.export(audio_path, format="mp3")
            logger.info(f"Audio extracted and saved to: {audio_path}")
        except Exception as e:
            logger.error(f"Failed to extract audio from video: {e}")
            raise Exception(f"Failed to extract audio from video: {e}")

        try:
            logger.info(f"Removing temporary video file: {video_file}")
            os.remove(video_file)
            logger.info(f"Temporary video file {video_file} removed.")
        except Exception as e:
            logger.warning(f"Failed to remove temporary video file {video_file}: {e}")
    else:
        if not os.path.exists(audio_path):
            raise Exception(f"Failed to download audio: {audio_path}")
        logger.info(f"Audio downloaded successfully: {audio_path}")

# Read from stream line by line until EOF, call callback on each line.
async def read_stream(stream, callback):
    while True:
        line = await stream.readline()
        if line:
            callback(line.decode())
        else:
            break

# Log each line from the stdout.
def log_stdout(line):
    logger.info(f"Whisper stdout: {line.strip()}")

# Log each line from the stderr.
def log_stderr(line):
    logger.error(f"Whisper stderr: {line.strip()}")

# transcription logic with header inclusion based on settings
# (always tries to use the gpu that's available with most free VRAM)
# transcribe_audio function
async def transcribe_audio(bot, update, audio_path, output_dir, youtube_url, video_info_message, include_header, model, device, language):
    log_gpu_utilization()  # Log GPU utilization before starting transcription

    logger.info(f"Using device: {device} for transcription")
    
    # transcription_command = ["whisper", audio_path, "--model", model, "--output_dir", output_dir, "--device", device]

    transcription_command = [
        "whisper", audio_path, 
        "--model", model, 
        "--output_dir", output_dir, 
        "--device", device
    ]

    if language and language != "auto":
        logger.info(f"Starting transcription with model '{model}' and language '{language}' for: {audio_path}")
        transcription_command.extend(["--language", language])
    else:
        logger.info(f"Starting transcription with model '{model}' and autodetect language for: {audio_path}")

    # Log the transcription command
    logger.info(f"Transcription command: {' '.join(transcription_command)}")

    try:
        # Start the subprocess and get stdout, stderr streams
        process = await asyncio.create_subprocess_exec(
            *transcription_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Concurrently log stdout and stderr
        await asyncio.gather(
            read_stream(process.stdout, log_stdout),
            read_stream(process.stderr, log_stderr)
        )

        # Wait for the subprocess to finish
        await process.wait()

        if process.returncode != 0:
            logger.error(f"Whisper process failed with return code {process.returncode}")
            return {}, ""

        logger.info(f"Whisper transcription completed for: {audio_path}")

        # Generate the header if needed, now including the model used
        ai_transcript_header = f"[ Transcript generated with: https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/ | OpenAI Whisper model: `{model}` | Language: `{language}` ]"
        header_content = ""

        if include_header:
            # Combine the video info message with the AI-generated transcript notice
            header_content = f"{video_info_message}\n\n{ai_transcript_header}\n\n"

        # Verify and log the generated files, adding header to .txt file if necessary
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        created_files = {}
        raw_content = ""

        for fmt in ['txt', 'srt', 'vtt']:
            file_path = f"{output_dir}/{base_filename}.{fmt}"
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                if fmt == 'txt':
                    with open(file_path, 'r') as f:
                        raw_content = f.read()
                    if include_header:
                        # Prepend the header for txt file
                        try:
                            with open(file_path, 'r') as original:
                                data = original.read()
                            with open(file_path, 'w') as modified:
                                modified.write(header_content + data)
                        except Exception as e:
                            logger.error(f"Error adding header to {file_path}: {e}")

                created_files[fmt] = file_path
                logger.info(f"Transcription file {'updated' if fmt == 'txt' and include_header else 'created'}: {file_path}")

        # Return created files and raw content for further processing
        return created_files, raw_content

    except Exception as e:
        logger.error(f"An error occurred during transcription: {e}")
        return {}, ""


# Process the message's URL and keep the user informed
# (Added in the new GPU logging function call to the process_url_message function)
async def process_url_message(message_text, bot, update, model, language):
    try:
        # Get transcription settings
        transcription_settings = get_transcription_settings()
        
        logger.info(f"Transcription settings in process_url_message: {transcription_settings}")

        user_id = update.effective_user.id
        urls = re.findall(r'(https?://\S+)', message_text)

        # Get the allowallsites setting from the configuration
        config = ConfigLoader.get_config()
        allow_all_sites = config.getboolean('GeneralSettings', 'allowallsites', fallback=False)

        for url in urls:

            if not allow_all_sites and not ("youtube" in url or "youtu.be" in url):
                await bot.send_message(chat_id=update.effective_chat.id, text="❌ Unsupported URL format. Currently, only YouTube URLs are fully supported.")
                continue

            # Normalize YouTube URL if it's a YouTube URL
            if "youtube" in url or "youtu.be" in url:
                normalized_url = normalize_youtube_url(url)
                if not normalized_url:
                    await bot.send_message(chat_id=update.effective_chat.id, text="Invalid YouTube URL.")
                    continue
            else:
                # For non-YouTube URLs, use the URL directly
                normalized_url = url

            logger.info(f"User {user_id} requested a transcript for normalized URL: {normalized_url}")
            await bot.send_message(chat_id=update.effective_chat.id, text="🔄 Processing URL...")

            audio_file_name = f"{user_id}_{int(time.time())}.mp3"
            audio_path = os.path.join(audio_dir, audio_file_name)
            video_info_message = "Transcription initiated."

            # Wrap fetch_video_details in try-except
            try:
                logger.info("Fetching video details...")
                details = await fetch_video_details(normalized_url)
                details['video_url'] = normalized_url
                video_info_message = create_video_info_message(details)
                for part in split_message(video_info_message):
                    await bot.send_message(chat_id=update.effective_chat.id, text=f"<code>{part}</code>", parse_mode='HTML')
            except Exception as e:
                error_message = str(e)
                logger.error(f"An error occurred while fetching video details: {error_message}")
                # await bot.send_message(chat_id=update.effective_chat.id, text=f"❌ Error: {error_message}")
                await bot.send_message(
                    chat_id=update.effective_chat.id, 
                    text=f"❌ Error: {error_message}", 
                    disable_web_page_preview=True
                )
                continue  # Skip to the next URL if any

            await bot.send_message(chat_id=update.effective_chat.id, text="📥 Fetching the audio track...")

            # Wrap download_audio in try-except
            try:
                await download_audio(normalized_url, audio_path)
            except Exception as e:
                # error_message = str(e)
                # await bot.send_message(chat_id=update.effective_chat.id, text=f"Error: {error_message}")
                # logger.error(f"Download audio failed for URL: {normalized_url}, error: {error_message}")

                error_message = str(e)
                # Truncate error_message if it's too long
                max_message_length = 4000  # Adjust as needed
                if len(error_message) > max_message_length:
                    error_message = error_message[:max_message_length] + '...'
                await bot.send_message(chat_id=update.effective_chat.id, text=f"Error: {error_message}")
                logger.error(f"Download audio failed for URL: {normalized_url}, error: {error_message}")

                continue

            if not os.path.exists(audio_path):
                logger.info(f"Audio download failed for URL: {normalized_url}")
                await bot.send_message(chat_id=update.effective_chat.id, text="❌ Failed to download audio. Please ensure the URL is correct and points to a supported video.")
                continue
            
            # Add this line to notify the user
            await bot.send_message(chat_id=update.effective_chat.id, text="✅ Audio download successful. Proceeding with transcription...")

            model = get_whisper_model(user_id)
            audio_duration = details.get('audio_duration', 0)
            estimated_time = estimate_transcription_time(model, audio_duration)
            estimated_minutes = estimated_time / 60
            current_time = datetime.now()
            estimated_finish_time = current_time + timedelta(minutes=estimated_minutes)

            time_now_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            estimated_finish_time_str = estimated_finish_time.strftime('%Y-%m-%d %H:%M:%S')

            log_message = (
                f"User ID: {user_id}\n"
                f"Requested URL for transcription: {normalized_url}\n"
            )

            best_gpu = get_best_gpu()
            if best_gpu:
                device = f'cuda:{best_gpu.id}'
                gpu_message = (
                    f"Using GPU {best_gpu.id}: {best_gpu.name}\n"
                    f"Free Memory: {best_gpu.memoryFree} MB\n"
                    f"Load: {best_gpu.load * 100:.1f}%"
                )
            else:
                device = 'cpu'
                gpu_message = "⚠️ WARNING: No CUDA GPU available, using CPU for transcription. This will be much slower than estimated."

            logger.info(gpu_message)
            await bot.send_message(chat_id=update.effective_chat.id, text=gpu_message)

            language_setting = language if language else "autodetection"
            detailed_message = (
                f"Whisper model in use:\n{model}\n\n"
                f"Model language set to:\n{language_setting}\n\n"
                f"Estimated transcription time:\n{estimated_minutes:.1f} minutes.\n\n"
                f"Time now:\n{time_now_str}\n\n"
                f"Time when finished (estimate):\n{estimated_finish_time_str}\n\n"
                "🎙️✍️ Transcribing audio..."
            )

            logger.info(f"{log_message}")
            logger.info(f"{detailed_message}")

            await bot.send_message(chat_id=update.effective_chat.id, text=detailed_message)

            transcription_paths, raw_content = await transcribe_audio(
                bot, update, audio_path, output_dir, normalized_url, video_info_message,
                transcription_settings['include_header'], model, device, language
            )

            if not transcription_paths:
                await bot.send_message(chat_id=update.effective_chat.id, text="Failed to transcribe audio.")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                continue

            # Add debugging here to see the settings at this point
            logger.info(f"send_as_messages setting before condition check: {transcription_settings['send_as_messages']}")

            # Here is where we add the transcription_note
            transcription_note = "📝🔊 <i>(transcribed audio)</i>\n\n"
            # note_length = len(transcription_note)
            # max_message_length = 4000 - note_length  # Adjust max length to account for transcription note

            # message sending and chunking logic; revised
            if transcription_settings['send_as_messages'] and 'txt' in transcription_paths:
                try:
                    logger.info(f"Preparing to send plain text message from raw content")
                    content = transcription_note + raw_content  # Add transcription note to the raw content
                    
                    # Just to be safe, reduce the chunk even more if needed
                    safe_max = 3000  # even safer limit

                    for i in range(0, len(content), safe_max):
                        chunk = content[i:i+safe_max]

                        # Optional: Make sure chunk length is safely under 4096 (should already be)
                        if len(chunk) > 4000:
                            chunk = chunk[:4000]

                        # # OPTIONAL: Check if we end in the middle of an HTML tag and adjust if needed.
                        # # For example, if chunk ends with '<', we might remove that character or find the previous space:
                        # if '<' in chunk[-5:]:  # crude check for partial tag at end
                        #     # Try to backtrack to a space before the '<'
                        #     last_space = chunk.rfind(' ')
                        #     if last_space != -1:
                        #         chunk = chunk[:last_space]

                        # Check if we end on a partial HTML tag
                        if '<' in chunk[-5:]:  # crude check for partial tag at end
                            last_space = chunk.rfind(' ')
                            if last_space != -1:
                                chunk = chunk[:last_space]
                        
                        # Attempt to find last whitespace so we don’t split in the middle of a word
                        # BUT if there's NO whitespace at all, we forcibly break anyway:
                        last_space = chunk.rfind(' ')
                        if last_space == -1 and len(chunk) == safe_max:
                            # This means no space was found in the entire chunk,
                            # so we forcibly keep the chunk at safe_max (which likely breaks a word).
                            logger.warning("No whitespace found. Forcibly splitting mid-word at position %d.", i + safe_max)
                            # chunk remains chunk[:safe_max], i.e. as-is
                        elif last_space > -1 and last_space > 0:
                            # We found a space within the chunk
                            chunk = chunk[:last_space]
                            # Note: If you do this, you might want to adjust `i` accordingly 
                            # or treat the leftover text on the next iteration. 
                            # But for a simple approach, this is enough to keep the code short.

                        # Now send the message
                        await bot.send_message(chat_id=update.effective_chat.id, text=chunk, parse_mode='HTML')
                        logger.info(f"Sent message chunk: {(i // safe_max) + 1}")
                except Exception as e:
                    logger.error(f"Error in sending plain text message: {e}")
            else:
                logger.info("Condition for sending plain text message not met.")
            
            # // old method
            # if transcription_settings['send_as_messages'] and 'txt' in transcription_paths:
            #     try:
            #         logger.info(f"Preparing to send plain text message from raw content")
            #         content = transcription_note + raw_content  # Add transcription note to the raw content
            #         for i in range(0, len(content), max_message_length):
            #             await bot.send_message(chat_id=update.effective_chat.id, text=content[i:i+max_message_length], parse_mode='HTML')
            #             logger.info(f"Sent message chunk: {i // max_message_length + 1}")
            #     except Exception as e:
            #         logger.error(f"Error in sending plain text message: {e}")
            # else:
            #     logger.info("Condition for sending plain text message not met.")

            # Sending files if configured
            if transcription_settings['send_as_files']:
                for fmt, path in transcription_paths.items():
                    try:
                        with open(path, 'rb') as file:
                            await bot.send_document(chat_id=update.effective_chat.id, document=file)
                        logger.info(f"Sent {fmt} file to user {update.effective_chat.id}: {path}")
                    except Exception as e:
                        logger.error(f"Failed to send {fmt} file to user {update.effective_chat.id}: {path}, error: {e}")

            # Check if we're keeping files or not
            if not transcription_settings['keep_audio_files'] and os.path.exists(audio_path):
                os.remove(audio_path)
            
            completion_log_message = f"Translation complete for user {user_id}, video: {normalized_url}, model: {model}"
            logging.info(completion_log_message)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        await bot.send_message(chat_id=update.effective_chat.id, text="An error occurred during processing.")

# create video info
def create_video_info_message(details):
    header_separator = "=" * 10
    video_info_message = f"""{header_separator}
Title: {details.get('title', 'No title available')}
Duration: {details.get('duration', 'No duration available')}
Channel: {details.get('channel', 'No channel information available')}
Upload Date: {details.get('upload_date', 'No upload date available')}
Views: {details.get('views', 'No views available')}
Likes: {details.get('likes', 'No likes available')}
Average Rating: {details.get('average_rating', 'No rating available')}
Comment Count: {details.get('comment_count', 'No comment count available')}
Channel ID: {details.get('channel_id', 'No channel ID available')}
Video ID: {details.get('video_id', 'No video ID available')}
Video URL: {details.get('video_url', 'No video URL available')}
Tags: {', '.join(details.get('tags', []) if isinstance(details.get('tags'), list) else ['No tags available'])}
Description: {details.get('description', 'No description available')}
{header_separator}"""
    return video_info_message   

# alt; shorten at 1000 chars.
# Description: {textwrap.shorten(details.get('description', 'No description available'), 1000, placeholder="...")}

# Helper function to format duration from seconds to H:M:S
def format_duration(duration):
    if not duration:
        return 'No duration available'
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"

# Fetch details for videos
async def fetch_video_details(url, max_retries=3, base_delay=5, command_timeout=30):
    command = ["yt-dlp", "--user-agent",
               "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
               "--dump-json", url]

    last_stderr_output = ""

    for attempt in range(max_retries):
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                # Set a timeout for the command execution
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=command_timeout)
            except asyncio.TimeoutError:
                logger.error(f"yt-dlp command timed out after {command_timeout} seconds")
                process.kill()
                await process.wait()
                stdout, stderr = None, b"Command timed out"

            if stderr and process.returncode != 0:
                stderr_output = stderr.decode()
                last_stderr_output = stderr_output  # Save the last stderr output
                logger.warning(f"Attempt {attempt + 1} failed: {stderr_output}")
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    logger.info(f"Retrying after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed.")
                    # Check for specific error messages
                    if any(keyword in last_stderr_output for keyword in [
                        "Sign in to confirm you're not a bot",
                        "unable to extract initial player response",
                        "This video is unavailable",
                        "Error 403",
                        "ERROR:"
                    ]):
                        custom_error_message = (
                            "❌ Failed to fetch video details due to YouTube's anti-bot measures or video restrictions. "
                            "Possible reasons include age restrictions, region locks, or the video requiring sign-in.\n"
                            "\nPlease try a different video URL, or see type /help for supported file formats for delivery.\n"
                            "\n⚠️ If you are the administrator of this service, consider using cookies with yt-dlp. "
                            "More info on yt-dlp's cookies at: https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
                        )
                        raise Exception(custom_error_message)
                    else:
                        raise Exception(f"Failed to fetch video details: {last_stderr_output}")
            else:
                try:
                    video_details = json.loads(stdout.decode()) if stdout else {}
                    return process_video_details(video_details, url)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from yt-dlp output: {e}")
                    raise Exception(f"❌ Error decoding JSON from yt-dlp output: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            # If this was the last attempt, re-raise the exception
            if attempt >= max_retries - 1:
                raise
            else:
                # Otherwise, log and retry
                wait_time = base_delay * (2 ** attempt)
                logger.info(f"Retrying after {wait_time} seconds...")
                await asyncio.sleep(wait_time)

# process the video details for included information
def process_video_details(video_details, url):
    description_text = video_details.get('description', '')
    if USE_SNIPPET_FOR_DESCRIPTION and description_text:
        description_text = get_description_snippet(description_text)

    # Extract duration directly and keep it in seconds
    audio_duration = int(video_details.get('duration', 0))

    tags = video_details.get('tags', [])

    # Check if tags is a list and it's not empty; otherwise, set a default message
    if isinstance(tags, list) and tags:
        tags_display = ', '.join(tags)
    else:
        tags_display = 'No tags available'

    return {
        'title': video_details.get('title', 'No title available'),
        'duration': format_duration(video_details.get('duration', 0)),
        'channel': video_details.get('uploader', 'No channel information available'),
        'upload_date': video_details.get('upload_date', 'No upload date available'),
        'views': video_details.get('view_count', 'No views available'),
        'likes': video_details.get('like_count', 'No likes available'),
        'average_rating': str(video_details.get('average_rating', 'No rating available')),
        'comment_count': str(video_details.get('comment_count', 'No comment count available')),
        'channel_id': video_details.get('channel_id', 'No channel ID available'),
        'video_id': video_details.get('id', 'No video ID available'),
        'video_url': video_details.get('webpage_url', url),
        'tags': tags_display,
        'description': description_text,
        'audio_duration': int(video_details.get('duration', 0))      
    }

# Helper function to get up to n lines from the description
def get_description_snippet(description, max_lines=DESCRIPTION_MAX_LINES):
    lines = description.split('\n')
    snippet = '\n'.join(lines[:max_lines])
    return snippet


# Regular expression for extracting the YouTube video ID
YOUTUBE_REGEX = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'


def extract_youtube_video_id(url):
    match = re.match(YOUTUBE_REGEX, url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(6)

def normalize_youtube_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    video_id = None

    if 'youtu.be' in parsed_url.netloc:
        # Extract video ID from path
        video_id = parsed_url.path.strip('/')
    elif 'youtube.com' in parsed_url.netloc:
        if 'v' in query_params:
            video_id = query_params['v'][0]
        else:
            # Handle URLs like https://www.youtube.com/embed/VIDEO_ID
            path_parts = parsed_url.path.split('/')
            if 'embed' in path_parts:
                embed_index = path_parts.index('embed')
                if len(path_parts) > embed_index + 1:
                    video_id = path_parts[embed_index + 1]
            elif 'shorts' in path_parts:
                shorts_index = path_parts.index('shorts')
                if len(path_parts) > shorts_index + 1:
                    video_id = path_parts[shorts_index + 1]
            elif len(path_parts) > 1:
                # For URLs like https://www.youtube.com/watch/VIDEO_ID
                video_id = path_parts[-1]
    else:
        logger.error(f"Unsupported YouTube URL format: {url}")
        return None

    if video_id:
        # Remove any additional parameters from the video ID
        video_id = video_id.split('?')[0].split('&')[0]
        # Construct the normalized URL
        return f'https://www.youtube.com/watch?v={video_id}'
    else:
        logger.error(f"Could not extract video ID from URL: {url}")
        return None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate transcription times
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define a dictionary to map models to their relative speeds
model_speeds = {
    'tiny': 32,
    'tiny.en': 32,
    'base': 16,
    'base.en': 16,
    'small': 6,
    'small.en': 6,
    'medium': 2,
    'medium.en': 2,
    'large': 1,
    'large-v1': 1,
    'large-v2': 1,
    'large-v3': 1,
    'turbo': 8
}

def estimate_transcription_time(model, audio_duration):
    """
    Estimate the transcription time based on the model size and audio duration.

    :param model: The model size used for transcription.
    :param audio_duration: The duration of the audio in seconds.
    :return: Estimated time in seconds to transcribe the audio.
    """
    # Ensure audio_duration is not None and is greater than 0
    if audio_duration is None or audio_duration <= 0:
        logger.error(f"Invalid audio duration: {audio_duration}")
        return 0
    
    logger.info(f"Estimating transcription time for model: {model} and audio duration: {audio_duration} seconds")
    
    # Assume 'large' model takes its duration equal to the audio's duration to transcribe.
    # Scale other models based on their relative speed.
    baseline_time = audio_duration  # This is for the 'large' model as a baseline
    relative_speed = model_speeds.get(model, 1)  # Default to 1 if model not found
    estimated_time = baseline_time / relative_speed
    
    logger.info(f"Estimated transcription time: {estimated_time} seconds")
    return max(estimated_time, 60)  # Ensure at least 1 minute is shown

# def estimate_transcription_time(model, audio_duration):
#     """
#     Estimate the transcription time based on the model size and audio duration.

#     :param model: The model size used for transcription.
#     :param audio_duration: The duration of the audio in seconds.
#     :return: Estimated time in seconds to transcribe the audio.
#     """
#     # Assume 'large' model takes its duration equal to the audio's duration to transcribe.
#     # Scale other models based on their relative speed.
#     baseline_time = audio_duration  # This is for the 'large' model as a baseline
#     relative_speed = model_speeds.get(model, 1)  # Default to 1 if model not found
#     estimated_time = baseline_time / relative_speed
#     return estimated_time

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get the best GPU availability
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Function to get GPU utilization and select the GPU with the most free memory
def get_best_gpu():
    gpus = GPUtil.getGPUs()
    if not gpus:
        logger.error("No GPUs found")
        return None  # Return None instead of 'cpu'
    
    best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)
    return best_gpu if best_gpu.memoryFree > 0 else None  # Return None if no GPU has free memory

# Add a new function to log GPU utilization details
def log_gpu_utilization():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        logger.info(f"GPU {gpu.id}: {gpu.name}, Load: {gpu.load * 100:.1f}%, Free Memory: {gpu.memoryFree} MB, Used Memory: {gpu.memoryUsed} MB, Total Memory: {gpu.memoryTotal} MB")
