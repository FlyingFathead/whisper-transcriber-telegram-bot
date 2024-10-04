# main.py
# ~~~~~~~
# openai-whisper transcriber-bot for Telegram

# version of this program
version_number = "0.1705"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import html
import time
import re
import signal
import asyncio
import logging
import configparser
import os
import subprocess
import datetime
from datetime import datetime, timedelta
from collections import defaultdict

from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CallbackContext
from telegram.ext import CommandHandler
from telegram.ext.filters import MessageFilter

# Adjust import paths based on new structure
from transcription_handler import process_url_message, set_user_model, get_whisper_model, transcribe_audio, get_best_gpu, get_audio_duration, estimate_transcription_time, format_duration, get_whisper_language, set_user_language
from utils.bot_token import get_bot_token
from utils.utils import print_startup_message, safe_split_message
from config_loader import ConfigLoader  # Import ConfigLoader

# Configure basic logging
logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Read configuration for restart behavior
# config = configparser.ConfigParser()
# config.read('config/config.ini')

# Use ConfigLoader to get configuration
config = ConfigLoader.get_config()
notification_settings = ConfigLoader.get_notification_settings()
restart_on_failure = config.getboolean('GeneralSettings', 'RestartOnConnectionFailure', fallback=True)

# Define directories for storing audio messages and files
audio_messages_dir = "audio_messages"
os.makedirs(audio_messages_dir, exist_ok=True)

# Initialize the lock outside of your function to ensure it's shared across all invocations.
queue_lock = asyncio.Lock()

class AllowedFileFilter(filters.MessageFilter):
    def __init__(self, allowed_formats):
        super().__init__()
        self.allowed_extensions = [fmt.lower().strip() for fmt in allowed_formats]

    def filter(self, message):
        if message.document:
            file_extension = message.document.file_name.split('.')[-1].lower()
            return file_extension in self.allowed_extensions
        return False

class TranscriberBot:

    # version of this program
    version_number = version_number

    # Class-level attribute for global locking
    processing_lock = asyncio.Lock() 

    def __init__(self):
        self.start_time = datetime.now()        

        self.token = get_bot_token()
        self.task_queue = asyncio.Queue()  # queue tasks
        self.is_processing = asyncio.Lock()  # Lock to ensure one transcription at a time

        self.restart_on_failure = restart_on_failure  # Controls the restart behavior on connection failure

        # self.config = configparser.ConfigParser()
        # self.config.read('config/config.ini')
        self.config = config  # Use the config from ConfigLoader        

        # get cooldown settings
        self.cooldown_seconds = self.config.getint('RateLimitSettings', 'cooldown_seconds', fallback=10)
        self.max_requests_per_minute = self.config.getint('RateLimitSettings', 'max_requests_per_minute', fallback=5)
        self.user_last_request = defaultdict(lambda: datetime.min)
        self.user_request_counts = defaultdict(int)

        # Load the allowed formats from the configuration
        self.allowed_formats = self.config.get('AllowedFileFormats', 'allowed_formats', fallback='mp3, wav, mp4').split(',')
        self.allowed_formats = [fmt.lower().strip() for fmt in self.allowed_formats]

        self.model = self.config.get('WhisperSettings', 'Model', fallback='medium.en')
        self.valid_models = self.config.get('ModelSettings', 'ValidModels', fallback='tiny, base, small, medium, large, turbo').split(', ')

        self.transcription_settings = {
            'includeheaderintranscription': self.config.getboolean('TranscriptionSettings', 'includeheaderintranscription', fallback=True),
            'keepaudiofiles': self.config.getboolean('TranscriptionSettings', 'keepaudiofiles', fallback=False),
            'sendasfiles': self.config.getboolean('TranscriptionSettings', 'sendasfiles', fallback=True),
            'sendasmessages': self.config.getboolean('TranscriptionSettings', 'sendasmessages', fallback=False)
        }

        logger.info(f"Transcription settings loaded: {self.transcription_settings}")

        self.model_change_limits = {}  # Dictionary to track user rate limits
        self.model_change_cooldown = 20  # Cooldown period in seconds
        self.user_models = {} # Use a dictionary to manage models per user.
        self.user_models_lock = asyncio.Lock()  # Lock for handling user_models dictionary

        # Define output directory for transcriptions
        self.output_dir = "transcriptions"
        os.makedirs(self.output_dir, exist_ok=True)

    async def start_command(self, update: Update, context: CallbackContext) -> None:
        welcome_message = (
            "👋 <b>Welcome to the Whisper Transcriber Bot!</b>\n\n"
            "I'm here to transcribe audio from various sources for you.\n\n"
            "📌 <b>How Does This Work?</b>\n"
            "- Send me a link to a supported media URL (e.g., YouTube).\n"
            "- Or, send an audio file (max 20MB in size), and I'll transcribe it.\n\n"
            "💡 <b>Commands You Can Use:</b>\n"
            "- /start: Show this welcome message.\n"
            "- /help: Get detailed help on how to use this service.\n"
            "- /info: View current settings and status.\n"
            "- /model: Change the transcription model.\n"
            "- /language: Set the transcription language.\n\n"
            "Let's get started! Send me a link or an audio file to begin."
        )
        await update.message.reply_text(welcome_message, parse_mode='HTML')

    async def handle_message(self, update: Update, context: CallbackContext) -> None:
        user_id = update.effective_user.id  # Update the user_id
        message_text = update.message.text  # Get the text of the message received

        # Log the received message along with the user ID
        logger.info(f"Received a message from user ID {user_id}: {message_text}")

        # Cooldown logic
        now = datetime.now()
        last_request_time = self.user_last_request[user_id]
        if (now - last_request_time).seconds < self.cooldown_seconds:
            await update.message.reply_text(f"Please wait {self.cooldown_seconds - (now - last_request_time).seconds} seconds before making another request.")
            return

        # Rate limiting logic
        minute_ago = now - timedelta(minutes=1)
        if self.user_request_counts[user_id] >= self.max_requests_per_minute and last_request_time > minute_ago:
            await update.message.reply_text("You have reached the maximum number of requests per minute. Please try again later.")
            return

        # Update request count and last request time
        if last_request_time < minute_ago:
            self.user_request_counts[user_id] = 0  # Reset the count if the last request was over a minute ago

        self.user_request_counts[user_id] += 1
        self.user_last_request[user_id] = now

        # Check and log the model before starting transcription
        current_model = get_whisper_model(user_id)
        logger.debug(f"Current model for user {user_id} before transcription: {current_model}")

        if update.message and update.message.text:
            urls = re.findall(r'(https?://\S+)', update.message.text)

            if urls:
                await self.task_queue.put((update.message.text, context.bot, update))
                queue_length = self.task_queue.qsize()

                logger.info(f"Task added to the queue. Current queue size: {queue_length}")

                # Check if this is the only job and nothing is currently processing.
                response_text = "⏳ Your request is next and is currently being processed." if queue_length == 1 else f"Your request has been added to the queue. There are {queue_length - 1} jobs ahead of yours."
                await update.message.reply_text(response_text)
            else:
                await update.message.reply_text("❌ No valid URL detected in your message. Please send a message that includes a valid URL. If you need help, type: /help")

    # async def process_queue(self):
    #     while True:
    #         message_text, bot, update = await self.task_queue.get()
    #         async with TranscriberBot.processing_lock:  # Use the class-level lock
    #             user_id = update.effective_user.id
    #             model = get_whisper_model(user_id)
    #             await process_url_message(message_text, bot, update, model)
    #         self.task_queue.task_done()

    async def process_queue(self):
        while True:
            try:
                task, bot, update = await self.task_queue.get()
                user_id = update.effective_user.id
                logger.info(f"Processing task for user ID {user_id}: {task}")

                try:
                    async with TranscriberBot.processing_lock:
                        success = False  # Flag to track if the task was successful
                        try:
                            model = get_whisper_model(user_id)
                            language = get_whisper_language(user_id)

                            if language == "auto":
                                language = None

                            video_info_message = ""  # Set a default empty value for video_info_message
                            ai_transcript_header = ""  # Set a default empty value for ai_transcript_header
                            transcription_note = "📝🔊 <i>(transcribed audio)</i>\n\n"  # Define transcription note

                            if isinstance(task, str) and task.startswith('http'):
                                logger.info(f"Processing URL: {task}")
                                await process_url_message(task, bot, update, model, language)
                                success = True  # Mark as successful if no exception occurs
                            elif any(task.endswith(f'.{ext}') for ext in self.allowed_formats):
                                logger.info(f"Processing audio/video file: {task}")

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
                                    gpu_message = "No GPU available, using CPU for transcription."

                                logger.info(gpu_message)
                                try:
                                    await bot.send_message(chat_id=update.effective_chat.id, text=gpu_message)
                                except Exception as e:
                                    logger.error(f"Failed to send GPU message: {e}")

                                audio_duration = get_audio_duration(task)
                                if audio_duration is None:
                                    try:
                                        await bot.send_message(chat_id=update.effective_chat.id, text="Invalid audio file. Please upload or link to a valid audio file.")
                                    except Exception as e:
                                        logger.error(f"Failed to send invalid audio file message: {e}")
                                    if os.path.exists(task):
                                        os.remove(task)
                                    continue

                                estimated_time = estimate_transcription_time(model, audio_duration)
                                estimated_minutes = estimated_time / 60  # Convert to minutes for user-friendly display

                                current_time = datetime.now()
                                estimated_finish_time = current_time + timedelta(seconds=estimated_time)

                                time_now_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
                                estimated_finish_time_str = estimated_finish_time.strftime('%Y-%m-%d %H:%M:%S')

                                formatted_audio_duration = format_duration(audio_duration)
                                language_setting = language if language else "autodetection"
                                detailed_message = (
                                    f"Audio file length:\n{formatted_audio_duration}\n\n"
                                    f"Whisper model in use:\n{model}\n\n"
                                    f"Model language set to:\n{language_setting}\n\n"
                                    f"Estimated transcription time:\n{estimated_minutes:.1f} minutes.\n\n"
                                    f"Time now:\n{time_now_str}\n\n"
                                    f"Time when finished (estimate):\n{estimated_finish_time_str}\n\n"
                                    "Transcribing audio..."
                                )
                                logger.info(detailed_message)
                                try:
                                    await bot.send_message(chat_id=update.effective_chat.id, text=detailed_message)
                                except Exception as e:
                                    logger.error(f"Failed to send detailed message: {e}")

                                transcription_paths, raw_content = await transcribe_audio(
                                    bot, update, task, self.output_dir, "", video_info_message,
                                    self.config.getboolean('TranscriptionSettings', 'includeheaderintranscription'),
                                    model, device, language
                                )

                                logger.info(f"Transcription paths returned: {transcription_paths}")

                                if not transcription_paths:
                                    try:
                                        await bot.send_message(chat_id=update.effective_chat.id, text="Failed to transcribe audio.")
                                    except Exception as e:
                                        logger.error(f"Failed to send failure message: {e}")
                                    if os.path.exists(task):
                                        os.remove(task)
                                    continue

                                # Send plain text as messages if configured to do so
                                if self.config.getboolean('TranscriptionSettings', 'sendasmessages') and 'txt' in transcription_paths:
                                    try:
                                        file_path = transcription_paths['txt']
                                        with open(file_path, 'r') as f:
                                            content = f.read()
                                            if self.config.getboolean('TranscriptionSettings', 'includeheaderintranscription'):
                                                ai_transcript_header = f"[ Transcript generated with: https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/ | OpenAI Whisper model: `{model}` | Language: `{language}` ]"
                                                header_content = f"{video_info_message}\n\n{ai_transcript_header}\n\n"
                                                content = content[len(header_content):]
                                            content = transcription_note + content  # Add transcription note

                                        # Escape content before splitting
                                        content = html.escape(content)

                                        # Define the maximum message length (Telegram limit is 4096)
                                        max_message_length = 3500

                                        # Split the content safely into chunks
                                        chunks = safe_split_message(content, max_length=max_message_length)

                                        # Send each chunk
                                        for i, chunk in enumerate(chunks):
                                            try:
                                                await bot.send_message(chat_id=update.effective_chat.id, text=chunk, parse_mode='HTML')
                                                logger.info(f"Sent message chunk: {i + 1}")
                                            except Exception as e:
                                                logger.error(f"Error sending message chunk {i + 1}: {e}")
                                                # Optionally, inform the user about the error
                                                try:
                                                    await bot.send_message(chat_id=update.effective_chat.id, text="An error occurred while sending a transcription chunk.")
                                                except Exception as e:
                                                    logger.error(f"Failed to send error message to user: {e}")
                                    except Exception as e:
                                        logger.error(f"Error in sending plain text messages: {e}")
                                        # Optionally, inform the user about the error
                                        try:
                                            await bot.send_message(chat_id=update.effective_chat.id, text="An error occurred while preparing the transcription messages.")
                                        except Exception as e:
                                            logger.error(f"Failed to send error message to user: {e}")
                                        # Continue to send files even if sending messages fails

                                # Proceed to send files as per your existing logic
                                if self.config.getboolean('TranscriptionSettings', 'sendasfiles'):
                                    for fmt, path in transcription_paths.items():
                                        try:
                                            with open(path, 'rb') as file:
                                                await bot.send_document(chat_id=update.effective_chat.id, document=file)
                                            logger.info(f"Sent {fmt} file to user {user_id}: {path}")
                                        except Exception as e:
                                            logger.error(f"Failed to send {fmt} file to user {user_id}: {path}, error: {e}")
                                            # Optionally, inform the user
                                            try:
                                                await bot.send_message(chat_id=update.effective_chat.id, text=f"An error occurred while sending the {fmt} file.")
                                            except Exception as e:
                                                logger.error(f"Failed to send error message to user: {e}")

                                # Mark the task as successful
                                success = True

                                # Clean up audio file if needed
                                if not self.config.getboolean('TranscriptionSettings', 'keepaudiofiles'):
                                    try:
                                        os.remove(task)
                                        logger.info(f"Deleted audio file: {task}")
                                    except Exception as e:
                                        logger.error(f"Failed to delete audio file {task}: {e}")

                        except Exception as e:
                            logger.error(f"An error occurred while processing the task: {e}")
                            # Optionally, inform the user about the error
                            try:
                                await bot.send_message(chat_id=update.effective_chat.id, text="An error occurred while processing your request.")
                            except Exception as ex:
                                logger.error(f"Failed to send error message to user: {ex}")

                        finally:
                            self.task_queue.task_done()

                            # Check if completion message should be sent
                            if success and notification_settings['send_completion_message']:
                                try:
                                    completion_message = notification_settings['completion_message']
                                    await bot.send_message(chat_id=update.effective_chat.id, text=completion_message)
                                    logger.info(f"Sent completion message to user ID {user_id}: {completion_message}")
                                except Exception as e:
                                    logger.error(f"Error sending completion message: {e}")

                except Exception as e:
                    logger.error(f"Unhandled exception in process_queue loop: {e}")
                    # Optionally, sleep briefly to prevent tight looping
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Unhandled exception in process_queue: {e}")
                # Prevent the loop from exiting
                await asyncio.sleep(0.1)

    async def shutdown(self, signal, loop):
        """Cleanup tasks tied to the service's shutdown."""
        logger.info(f"Received exit signal {signal.name}...")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

        [task.cancel() for task in tasks]

        logger.info(f"Cancelling {len(tasks)} outstanding tasks")
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    # set the model's language
    async def set_language_command(self, update: Update, context: CallbackContext) -> None:
        user_id = update.effective_user.id
        supported_languages = self.config.get('WhisperSettings', 'supportedlanguages', fallback='auto').split(', ')

        if not context.args:
            # Display the supported languages if no argument is provided
            await update.message.reply_text(
                f"Please specify a supported language code or set to <code>auto</code> for autodetect.\n\nExamples:\n<code>/language en</code>\n<code>/language auto</code>\n\n"
                f"Supported languages are: {', '.join(supported_languages)}",
                parse_mode='HTML'
            )
            return

        language_code = context.args[0]
        if language_code in supported_languages:
            set_user_language(user_id, language_code)
            await update.message.reply_text(f"Language set to: {language_code}")
        else:
            await update.message.reply_text(
                f"Unsupported language code. Supported languages are: {', '.join(supported_languages)}",
                parse_mode='HTML'
            )

    # view help
    async def help_command(self, update: Update, context: CallbackContext) -> None:
        models_list = ', '.join(self.valid_models)  # Dynamically generate the list of valid models
        allowed_formats_list = ', '.join(self.allowed_formats)  # Get the list of allowed formats

        # Access the 'allowaudiofiles' and 'allowvoicemessages' settings
        allow_audio_files = self.config.getboolean('AudioSettings', 'allowaudiofiles', fallback=True)
        allow_voice_messages = self.config.getboolean('AudioSettings', 'allowvoicemessages', fallback=True)

        # Build the file upload info based on settings
        file_upload_info = ""
        if allow_audio_files and allow_voice_messages:
            file_upload_info = (
                "- Or, send an audio message or an audio file to have its audio transcribed. (maximum file size: 20MB)\n\n"
                f"<b>Currently supported file formats:</b> {allowed_formats_list}\n"
            )
        elif allow_audio_files and not allow_voice_messages:
            file_upload_info = (
                "- Or, send an audio file to have its audio transcribed. (maximum file size: 20MB)\n\n"
                f"<b>Currently supported file formats:</b> {allowed_formats_list}\n"
            )
        elif not allow_audio_files and allow_voice_messages:
            file_upload_info = (
                "- Or, send an audio message to have its audio transcribed. (maximum file size: 20MB)\n"
                "- Note: Direct file uploads are currently disabled.\n"
            )
        else:
            file_upload_info = (
                "- Note: Direct file uploads and audio messages are currently disabled.\n"
            )

        help_text = f"""<b>Welcome to the Whisper Transcriber Bot!</b>

    <b>Version:</b> {self.version_number}

    <b>How to Use:</b>
    - Send any supported media URL to have its audio transcribed.
    {file_upload_info}
    - Use /info to view the current settings, status, and jobs in queue.
    - Use /model to change the transcription model.
    - Use /language to change the model language in use
    (set language to <code>auto</code> for automatic language detection).

    <i>TIP: Setting the language manually to the audio's language may improve accuracy and speed.</i>

    - Use /help or /about to display this help message.

    <b>Whisper model currently in use:</b>
    <code>{self.model}</code>

    <b>Available Whisper models:</b>
    {models_list}

    <b>Bot code by FlyingFathead.</b>
    Source code on <a href='https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/'>GitHub</a>.

    <b>Disclaimer:</b>
    The original author is NOT responsible for how this bot is utilized. All code and outputs are provided 'AS IS' without warranty of any kind. Users assume full responsibility for the operation and output of the bot. This applies to both legal and ethical responsibilities. Use at your own risk.
    """
        await update.message.reply_text(help_text, parse_mode='HTML')

    async def model_command(self, update: Update, context: CallbackContext) -> None:
        user_id = update.effective_user.id
        current_time = time.time()
        models_list = ', '.join(self.valid_models)  # Dynamically generate the list of valid models
        
        if not context.args:
            current_model = get_whisper_model(user_id)
            await update.message.reply_text(
                f"<b>Current model in use:</b>\n<code>{current_model}</code>\n\n"
                f"<b>Available models:</b>\n{models_list}\n\n"
                "To change the model, use commands like:\n"
                "<code>/model turbo</code> (newest model, 8x real-time)\n"
                "<code>/model medium.en</code>\n"
                "<code>/model large-v3</code> (most precise, but slowest)\n\n"
                "<b>Model Details:</b>\n"
                "- <b>Tiny</b>: Fastest, uses ~1GB VRAM, about 32x faster than large.\n"
                "- <b>Base</b>: Faster, uses ~1GB VRAM, about 16x faster than large.\n"
                "- <b>Small</b>: Balanced, uses ~2GB VRAM, about 6x faster than large.\n"
                "- <b>Medium</b>: More precise, uses ~5GB VRAM, about 2x faster than large.\n"
                "- <b>Large</b>: Most precise, processes at real-time (1x speed), uses ~10GB VRAM.\n"
                "- <b>Turbo (New!)</b>: Introduced in September 2024, near large-v3-level quality, 8x realtime speed, uses ~6GB VRAM. <b>Recommended by default.</b>\n\n"
                "Note: '.en' models (e.g., 'tiny.en') are optimized for English and offer better accuracy for English audio. "
                "As model size increases, the benefit of English optimization becomes less significant. Choose based on your "
                "needs for speed, memory usage, and linguistic accuracy. "
                "As a general guideline, larger models are more accurate but also slower. "
                "I.e. best balance in speed vs. accuracy in English language is likely either <code>turbo</code> or <code>medium.en</code>.",
                parse_mode='HTML')
            return

        new_model = context.args[0]
        if new_model in self.valid_models:
            set_user_model(user_id, new_model)  # Update user-specific model
            self.model_change_limits[user_id] = current_time  # Update cooldown tracker
            await update.message.reply_text(f"Model set to: <code>{new_model}</code>", parse_mode='HTML')
        else:
            await update.message.reply_text(
                f"Invalid model specified.\n\n"
                f"Available models:\n{models_list}",
                parse_mode='HTML')

    async def handle_voice_message(self, update: Update, context: CallbackContext) -> None:
        user_id = update.effective_user.id
        voice = update.message.voice
        
        if not self.config.getboolean('AudioSettings', 'allowvoicemessages'):
            await update.message.reply_text("Voice messages are not allowed.")
            return

        file = await context.bot.get_file(voice.file_id)
        ogg_file_path = os.path.join(audio_messages_dir, f'{file.file_id}.ogg')
        wav_file_path = os.path.join(audio_messages_dir, f'{file.file_id}.wav')
        await file.download_to_drive(ogg_file_path)
        
        # Convert Ogg Opus to WAV using ffmpeg
        try:
            subprocess.run(['ffmpeg', '-i', ogg_file_path, wav_file_path], check=True)
            logger.info(f"Converted voice message to WAV format: {wav_file_path}")
            
            await self.task_queue.put((wav_file_path, context.bot, update))
            queue_length = self.task_queue.qsize()
            response_text = "Your request is next and is currently being processed." if queue_length == 1 else f"Your request has been added to the queue. There are {queue_length - 1} jobs ahead of yours."
            await update.message.reply_text(response_text)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting voice message: {e}")

    async def handle_audio_file(self, update: Update, context: CallbackContext) -> None:
        logger.info("handle_audio_file called.")

        user_id = update.effective_user.id

        # Check if file uploads are allowed
        allow_audio_files = self.config.getboolean('AudioSettings', 'allowaudiofiles', fallback=True)
        logger.info(f"allow_audio_files: {allow_audio_files}")
        if not allow_audio_files:
            await update.message.reply_text("File processing is not allowed.")
            logger.info("File processing is not allowed according to config.")
            return

        # Determine whether the message contains a document or an audio
        document = update.message.document
        audio = update.message.audio

        if document:
            file_info = document
            file_name = document.file_name
            logger.info(f"Received a document from user {user_id}: {file_name}")
        elif audio:
            file_info = audio
            file_name = audio.file_name or f"{audio.file_unique_id}.mp3"
            logger.info(f"Received an audio file from user {user_id}: {file_name}")
        else:
            logger.info("No document or audio found in the message.")
            return

        try:
            # Check file size before downloading
            file_size = file_info.file_size
            if file_size > 20 * 1024 * 1024:  # 20 MB in bytes
                await update.message.reply_text(
                    "The file is too large to process. "
                    "Telegram bots can only download files up to 20 MB in size. "
                    "Please send a smaller file or provide a link to the audio."
                )
                logger.warning(f"File is too big: {file_size} bytes.")
                return

            file_extension = file_name.split('.')[-1].lower()
            logger.info(f"Extracted file extension: {file_extension}")
            logger.info(f"Allowed formats: {self.allowed_formats}")

            if file_extension not in self.allowed_formats:
                await update.message.reply_text(
                    f"Files with extension .{file_extension} are not supported.\n"
                    f"Supported formats are: {', '.join(self.allowed_formats)}"
                )
                logger.info("File extension not in allowed formats.")
                return

            # Proceed with downloading and processing the file
            file = await context.bot.get_file(file_info.file_id)
            file_path = os.path.join(audio_messages_dir, f'{file_info.file_unique_id}.{file_extension}')
            await file.download_to_drive(file_path)
            logger.info(f"File downloaded to {file_path}")

            # Queue the file for transcription
            await self.task_queue.put((file_path, context.bot, update))
            queue_length = self.task_queue.qsize()
            response_text = (
                "Your request is next and is currently being processed."
                if queue_length == 1
                else f"Your request has been added to the queue. There are {queue_length - 1} jobs ahead of yours."
            )
            await update.message.reply_text(response_text)
            logger.info(f"File queued for transcription. Queue length: {queue_length}")
        except Exception as e:
            logger.error(f"Exception in handle_audio_file: {e}")
            await update.message.reply_text("An error occurred while processing your file.")

    async def info_command(self, update: Update, context: CallbackContext) -> None:
        user_id = update.effective_user.id
        current_model = get_whisper_model(user_id)
        current_language = get_whisper_language(user_id)
        uptime = datetime.now() - self.start_time  # Assuming self.start_time is set at bot launch
        gpu_info = get_best_gpu()
        queue_length = self.task_queue.qsize()

        if gpu_info:
            gpu_status = f"GPU {gpu_info.id}: {gpu_info.name}, Free Memory: {gpu_info.memoryFree} MB"
        else:
            gpu_status = "No GPU available, using CPU"

        info_message = (
            f"<b>Current model in use:</b> {current_model}\n"
            f"(change with: /model)\n\n"
            f"<b>Selected transcription language:</b> {current_language}\n"
            f"(change with /language)\n\n"
            f"<b>Bot uptime:</b> {str(uptime)}\n"
            f"<b>Bot started on:</b> {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"<b>Current active GPU status:</b>\n{gpu_status}\n\n"
            f"<b>Jobs in queue:</b> {queue_length}"
        )

        await update.message.reply_text(info_message, parse_mode='HTML')

    # run
    def run(self):
        
        loop = asyncio.get_event_loop()

        for sig in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s, loop)))

        connected = False
        while not connected:
            try:
                self.application = Application.builder().token(self.token).build()

                # Add command handlers first
                self.application.add_handler(CommandHandler('start', self.start_command))                
                self.application.add_handler(CommandHandler(['help', 'about'], self.help_command))
                self.application.add_handler(CommandHandler('info', self.info_command))
                self.application.add_handler(CommandHandler('model', self.model_command))
                self.application.add_handler(CommandHandler('language', self.set_language_command))

                # Add specific message handlers next
                self.application.add_handler(MessageHandler(filters.AUDIO, self.handle_audio_file))
                self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice_message))
                self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_audio_file))

                # Add generic message handler last
                self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

                loop.create_task(self.process_queue())
                self.application.run_polling()
                connected = True

            except Exception as e:
                logger.error(f"Failed to start polling due to an error: {e}")
                if self.restart_on_failure:
                    logger.info("Attempting to reconnect in 10 seconds...")
                    time.sleep(10)
                else:
                    logger.error("Restart on failure is disabled. Exiting...")
                    break

if __name__ == '__main__':
    print_startup_message(version_number)  # Print startup message

    # Read the update settings from config
    config = ConfigLoader.get_config()
    check_for_updates = config.getboolean('UpdateSettings', 'CheckForYTDLPUpdates', fallback=False)
    update_command = config.get('UpdateSettings', 'UpdateCommand', fallback='pip install -U yt-dlp')

    if check_for_updates:
        logger.info("Checking for yt-dlp updates...")
        try:
            result = subprocess.run(
                update_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            logger.info(f"yt-dlp update output:\n{result.stdout.decode()}")
            logger.info("yt-dlp updated successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update yt-dlp: {e.stderr.decode()}")
        except Exception as e:
            logger.error(f"An error occurred while updating yt-dlp: {e}")

    bot = TranscriberBot()
    bot.run()
