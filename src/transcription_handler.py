# transcription_handler.py
# ~~~~~~~
# openai-whisper transcriber-bot for Telegram
# v0.04.1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import time
import logging
import re
import asyncio
import json
import os
import textwrap
import configparser
from urllib.parse import urlparse, parse_qs

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
    command = ["yt-dlp", "--extract-audio", "--audio-format", "mp3", url, "-o", output_path]

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

    # Prepare the full header content
    ai_transcript_header = "Whisper AI-generated transcript:"
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

            # Download the audio from the normalized URL
            await download_audio(normalized_url, audio_path)

            if not os.path.exists(audio_path):
                # Notify if audio download fails
                logger.info(f"Audio download failed for URL: {normalized_url}")
                await bot.send_message(chat_id=update.effective_chat.id, text="Failed to download audio. Please ensure the URL is correct and points to a supported video.")
                continue

            video_info_message = "Transcription initiated."

            # Fetch and process YouTube video details only if it's a YouTube URL
            # If it's a YouTube URL, fetch additional video details
            video_info_message = "Transcription initiated."
            if 'youtube.com' in normalized_url or 'youtu.be' in normalized_url:
                logger.info("Fetching YouTube video details...")
                details = await fetch_youtube_details(normalized_url)
                if details:
                    # Construct video information message
                    video_info_message = create_video_info_message(details)
                    await bot.send_message(chat_id=update.effective_chat.id, text=f"<code>{video_info_message}</code>", parse_mode='HTML')
                else:
                    logger.error("Failed to fetch YouTube video details.")
            
            # Inform the user that the transcription process has started
            await bot.send_message(chat_id=update.effective_chat.id, text="Transcribing audio... This may take some time.")

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
            await bot.send_message(chat_id=update.effective_chat.id, text="Transcription complete. Have a nice day!")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        await bot.send_message(chat_id=update.effective_chat.id, text="An error occurred during processing.")

# create video info
def create_video_info_message(details):
    header_separator = "=" * 30
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
Tags: {', '.join(details.get('tags', ['No tags available']))}
Description: {textwrap.shorten(details.get('description', 'No description available'), 1000, placeholder="...")}
{header_separator}"""
    return video_info_message   

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

# Fetch details for YouTube videos
async def fetch_youtube_details(url, max_retries=3, base_delay=5):
    command = ["yt-dlp", "--user-agent",
               "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
               "--dump-json", url]

    for attempt in range(max_retries):
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if stderr and process.returncode != 0:
            logger.warning(f"Attempt {attempt + 1} failed: {stderr.decode()}")
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying after {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error("All retry attempts failed.")
        else:
            try:
                video_details = json.loads(stdout.decode())
                duration_formatted = format_duration(video_details.get('duration', 0))                

                if USE_SNIPPET_FOR_DESCRIPTION:
                    description_text = get_description_snippet(video_details.get('description', 'No description available'))
                else:
                    description_text = video_details.get('description', 'No description available')

                # Directly use the passed URL for video_url
                filtered_details = {
                    'title': video_details.get('title', 'No title available'),
                    'duration': duration_formatted,
                    'channel': video_details.get('uploader', 'No channel information available'),
                    'upload_date': video_details.get('upload_date', 'No upload date available'),
                    'views': video_details.get('view_count', 'No views available'),
                    'likes': video_details.get('like_count', 'No likes available'),
                    'average_rating': video_details.get('average_rating', 'No rating available'),
                    'comment_count': video_details.get('comment_count', 'No comment count available'),
                    'channel_id': video_details.get('channel_id', 'No channel ID available'),
                    'video_id': video_details.get('id', 'No video ID available'),
                    'tags': video_details.get('tags', ['No tags available']),
                    'description': description_text,
                    'video_url': url  # Set this directly using the URL passed to the function
                }

                logger.info(f"Fetched YouTube details successfully for URL: {url}")
                return filtered_details
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from yt-dlp output: {e}")
                return None
    return None

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