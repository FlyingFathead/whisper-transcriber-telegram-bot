# transcription_handler.py
# ~~~
# openai-whisper transcriber-bot for Telegram
# v0.07.2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import time
import logging
import re
import asyncio
from asyncio.exceptions import TimeoutError
import json
import os
import textwrap
import configparser
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta

# internal modules
from utils.language_selection import ask_language

# tg modules // button selection
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# Toggle this to use the full description or a snippet.
USE_SNIPPET_FOR_DESCRIPTION = False

# If we're using a snippet of the description, maximum number of lines to include
DESCRIPTION_MAX_LINES = 30

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

# get the general settings
def get_general_settings():
    config = configparser.ConfigParser()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'config.ini')
    config.read(config_path)
    allow_all_sites = config.getboolean('GeneralSettings', 'AllowAllSites', fallback=False)
    return {
        'allow_all_sites': allow_all_sites
    }

# get whisper model
def get_whisper_model():
    config = configparser.ConfigParser()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'config.ini')
    config.read(config_path)
    model = config.get('WhisperSettings', 'Model', fallback='base')
    return model

# get transcription settings
def get_transcription_settings():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'config.ini')

    if not os.path.exists(config_path):
        logger.error("Error: config.ini not found at the expected path.")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_path)

    # Ensure 'TranscriptionSettings' section exists
    if 'TranscriptionSettings' not in config:
        logger.error("TranscriptionSettings section missing in config.ini")
        sys.exit(1)

    include_header = config.getboolean('TranscriptionSettings', 'IncludeHeaderInTranscription', fallback=False)
    keep_audio_files = config.getboolean('TranscriptionSettings', 'KeepAudioFiles', fallback=False)

    logger.info(f"Transcription settings loaded: include_header={include_header}, keep_audio_files={keep_audio_files}")
    
    return {
        'include_header': include_header,
        'keep_audio_files': keep_audio_files
    }

# audio download
async def download_audio(url, output_path):

    logger.info(f"Attempting to download audio from: {url}")
    
    # Specify a cache directory that yt-dlp can write to
    cache_dir = ".cache"

    # Check if the cache directory exists, create it if it doesn't
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory: {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directory {cache_dir}: {e}")
            # Optionally, handle the error (e.g., use a default cache dir or abort the operation)

    command = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--cache-dir", cache_dir,  # Specify the custom cache directory
        url,
        "-o", output_path
    ]

    # Start the subprocess and capture stdout and stderr
    process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    # Communicate with the process to retrieve its output and error message
    stdout, stderr = await process.communicate()

    # Log the stdout and stderr
    if stdout:
        logger.info(f"yt-dlp stdout: {stdout.decode().strip()}")
    if stderr:
        logger.error(f"yt-dlp stderr: {stderr.decode().strip()}")

    # Check if the file was downloaded successfully
    if os.path.exists(output_path):
        logger.info(f"Audio downloaded successfully: {output_path}")
    else:
        logger.error(f"Failed to download audio: {output_path}")

# transcription logic with header inclusion based on settings
async def transcribe_audio(audio_path, output_dir, youtube_url, video_info_message, include_header):

    # set the transcription command
    model = get_whisper_model()

    logger.info(f"Starting transcription with model '{model}' for: {audio_path}")

    transcription_command = ["whisper", audio_path, "--model", model, "--output_dir", output_dir]

    process = await asyncio.create_subprocess_exec(
        *transcription_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    # Check if Whisper process encountered an error
    if process.returncode != 0:
        logger.error(f"Whisper process failed with return code {process.returncode}")
        logger.error(f"Whisper STDERR: {stderr.decode()}")
        return {}

    logger.info(f"Whisper transcription completed for: {audio_path}")

    # Generate the header if needed
    header_content = ""

    # Generate the header if needed, now including the model used
    ai_transcript_header = f"[ Transcript generated by Whisper AI using the model: `{model}` ]"
    header_content = ""

    if include_header:
        # Combine the video info message with the AI-generated transcript notice
        header_content = f"{video_info_message}\n\n{ai_transcript_header}\n\n"

    # Verify and log the generated files, adding header to .txt file
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    created_files = {}

    for fmt in ['txt', 'srt', 'vtt']:
        file_path = f"{output_dir}/{base_filename}.{fmt}"
        if fmt == 'txt' and include_header and os.path.exists(file_path):
            # Prepend the header for txt file
            with open(file_path, 'r') as original: data = original.read()
            with open(file_path, 'w') as modified: modified.write(header_content + data)
            
        # Log the creation and update of each file
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            logger.info(f"Transcription file {'updated' if fmt == 'txt' and include_header else 'created'}: {file_path}")
            created_files[fmt] = file_path
        else:
            logger.warning(f"Expected transcription file not found or empty: {file_path}")

    return created_files

# Process the message's URL and keep the user informed
async def process_url_message(message_text, bot, update):

    try:

        # Get general settings right at the beginning of the function
        settings = get_general_settings()

        # Get general and transcription settings at the beginning of the function
        transcription_settings = get_transcription_settings()

        include_header = transcription_settings.get('include_header', False)
        keep_audio_files = transcription_settings.get('keep_audio_files', False)

        # Get user ID from the update object
        user_id = update.effective_user.id

        # Parse the url from the message text
        urls = re.findall(r'(https?://\S+)', message_text)

        for url in urls:
            # Normalize the YouTube URL to strip off any unnecessary parameters
            normalized_url = normalize_youtube_url(url)
            logger.info(f"User {user_id} requested a transcript for normalized URL: {normalized_url}")

            # Notify the user that the bot is processing the URL
            await bot.send_message(chat_id=update.effective_chat.id, text="Processing URL...")

            # Define audio file name and path
            audio_file_name = f"{user_id}_{int(time.time())}.mp3"
            audio_path = os.path.join(audio_dir, audio_file_name)

            video_info_message = "Transcription initiated."

            # Fetch video details
            logger.info("Fetching video details...")
            details = await fetch_video_details(normalized_url)
            if details:
                # Construct video information message
                # Pass the normalized URL directly into the video info creation
                details['video_url'] = normalized_url                
                video_info_message = create_video_info_message(details)
                await bot.send_message(chat_id=update.effective_chat.id, text=f"<code>{video_info_message}</code>", parse_mode='HTML')
            else:
                logger.error("Failed to fetch video details.")

            # Inform the user that the transcription process has started
            await bot.send_message(chat_id=update.effective_chat.id, text="Fetching the audio track...")

            # Download the audio from the normalized URL
            await download_audio(normalized_url, audio_path)

            if not os.path.exists(audio_path):
                # Notify if audio download fails
                logger.info(f"Audio download failed for URL: {normalized_url}")
                await bot.send_message(chat_id=update.effective_chat.id, text="Failed to download audio. Please ensure the URL is correct and points to a supported video.")
                continue
            
            # Inform the user that the transcription process has started and do a time estimate
            model = get_whisper_model()

            # Use the audio duration from the video details
            audio_duration = details['audio_duration']
            estimated_time = estimate_transcription_time(model, audio_duration)
            estimated_minutes = estimated_time / 60  # Convert to minutes for user-friendly display

            # Calculate estimated finish time
            current_time = datetime.now()
            estimated_finish_time = current_time + timedelta(minutes=estimated_minutes)

            # Format messages for start and estimated finish time
            time_now_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            estimated_finish_time_str = estimated_finish_time.strftime('%Y-%m-%d %H:%M:%S')

            # Detailed message for logging with user_id and requested URL
            log_message = (
                f"User ID: {user_id}\n"
                f"Requested URL for transcription: {normalized_url}\n"
            )

            # Prepare and send the detailed message
            detailed_message = (
                f"Whisper model in use: {model}\n\n"                
                f"Estimated transcription time: {estimated_minutes:.1f} minutes.\n\n"
                f"Time now:\n{time_now_str}\n\n"
                f"Time when finished (estimate):\n{estimated_finish_time_str}\n\n"
                "Transcribing audio..."
            )

            # Concatenate and log the messages for internal records
            logger.info(f"{log_message}")
            logger.info(f"{detailed_message}")

            await bot.send_message(chat_id=update.effective_chat.id, text=detailed_message)

            # Transcribe the audio and handle transcription output
            transcription_paths = await transcribe_audio(audio_path, output_dir, normalized_url, video_info_message, include_header)
            if not transcription_paths:
                # Notify if transcription fails
                await bot.send_message(chat_id=update.effective_chat.id, text="Failed to transcribe audio.")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                continue

            # Send transcription files and finalize the process
            for fmt, path in transcription_paths.items():
                await bot.send_document(chat_id=update.effective_chat.id, document=open(path, 'rb'))
            if not keep_audio_files and os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Log the completion message with user ID and video URL
            completion_log_message = f"Translation complete for user {user_id}, video: {normalized_url}"
            logging.info(completion_log_message)
            await bot.send_message(chat_id=update.effective_chat.id, text="Transcription complete. Have a nice day!")

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
        return f"{hours}h {minutes}m {seconds}s"
    else:
        return f"{minutes}m {seconds}s"

# Fetch details for videos
async def fetch_video_details(url, max_retries=3, base_delay=5, command_timeout=30):
    command = ["yt-dlp", "--user-agent",
               "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
               "--dump-json", url]

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
            except TimeoutError:
                logger.error(f"yt-dlp command timed out after {command_timeout} seconds")
                process.kill()
                await process.wait()
                stdout, stderr = None, b"Command timed out"

            if stderr and process.returncode != 0:
                logger.warning(f"Attempt {attempt + 1} failed: {stderr.decode()}")
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    logger.info(f"Retrying after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed.")
                    return None
            else:
                try:
                    video_details = json.loads(stdout.decode()) if stdout else {}
                    return process_video_details(video_details, url)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from yt-dlp output: {e}")
                    return None
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return None

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
YOUTUBE_REGEX = (
    r'(https?://)?(www\.)?'
    '(youtube|youtu|youtube-nocookie)\.(com|be)/'
    '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

def extract_youtube_video_id(url):
    match = re.match(YOUTUBE_REGEX, url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(6)

def normalize_youtube_url(url):
    # Parse the URL
    parsed_url = urlparse(url)
    # Check if there are any query parameters
    query_params = parse_qs(parsed_url.query)
    # If 'v' parameter is in the query, reconstruct the URL without any other parameters
    video_id = query_params.get('v')
    if video_id:
        return f'https://www.youtube.com/watch?v={video_id[0]}'
    # If there is no 'v' parameter, return the original URL (or handle accordingly)
    return url

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate transcription times
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define a dictionary to map models to their relative speeds
model_speeds = {
    'tiny': 32,
    'base': 16,
    'small': 6,
    'medium': 2,
    'large': 1
}

def estimate_transcription_time(model, audio_duration):
    """
    Estimate the transcription time based on the model size and audio duration.

    :param model: The model size used for transcription.
    :param audio_duration: The duration of the audio in seconds.
    :return: Estimated time in seconds to transcribe the audio.
    """
    # Assume 'large' model takes its duration equal to the audio's duration to transcribe.
    # Scale other models based on their relative speed.
    baseline_time = audio_duration  # This is for the 'large' model as a baseline
    relative_speed = model_speeds.get(model, 1)  # Default to 1 if model not found
    estimated_time = baseline_time / relative_speed
    return estimated_time
