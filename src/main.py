# main.py
# ~~~~~~~
# openai-whisper transcriber-bot for Telegram

# version of this program
version_number = "0.10"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import time
import re
import signal
import asyncio
import logging
import configparser

from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CallbackContext
from telegram.ext import CommandHandler

# Adjust import paths based on new structure
from transcription_handler import process_url_message
from utils.bot_token import get_bot_token
from utils.utils import print_startup_message

# Configure basic logging
logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Read configuration for restart behavior
config = configparser.ConfigParser()
config.read('config/config.ini')
restart_on_failure = config.getboolean('GeneralSettings', 'RestartOnConnectionFailure', fallback=True)

# Initialize the lock outside of your function to ensure it's shared across all invocations.
queue_lock = asyncio.Lock()

class TranscriberBot:

    # version of this program
    version_number = version_number

    def __init__(self):
        self.token = get_bot_token()
        self.task_queue = asyncio.Queue()  # queue tasks
        self.is_processing = asyncio.Lock()  # Lock to ensure one transcription at a time

        self.restart_on_failure = restart_on_failure  # Controls the restart behavior on connection failure

        self.config = configparser.ConfigParser()
        self.config.read('config/config.ini')
        self.model = self.config.get('WhisperSettings', 'Model', fallback='medium.en')
        self.valid_models = self.config.get('ModelSettings', 'ValidModels', fallback='tiny, base, small, medium, large').split(', ')

    async def handle_message(self, update: Update, context: CallbackContext) -> None:
        logger.info("Received a message.")
        if update.message and update.message.text:
            urls = re.findall(r'(https?://\S+)', update.message.text)

            if urls:
                await self.task_queue.put((update.message.text, context.bot, update))
                queue_length = self.task_queue.qsize()

                logger.info(f"Task added to the queue. Current queue size: {queue_length}")

                # Check if this is the only job and nothing is currently processing.
                response_text = "Your request is next and is currently being processed." if queue_length == 1 else f"Your request has been added to the queue. There are {queue_length - 1} jobs ahead of yours."
                await update.message.reply_text(response_text)
            else:
                await update.message.reply_text("No valid URL detected in your message. Please send a message that includes a valid URL. If you need help, type: /help")

    async def process_queue(self):
        while True:
            message_text, bot, update = await self.task_queue.get()
            async with self.is_processing:  # Ensure one job at a time
                await process_url_message(message_text, bot, update)
            self.task_queue.task_done()

    async def shutdown(self, signal, loop):
        """Cleanup tasks tied to the service's shutdown."""
        logger.info(f"Received exit signal {signal.name}...")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

        [task.cancel() for task in tasks]

        logger.info(f"Cancelling {len(tasks)} outstanding tasks")
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    async def help_command(self, update: Update, context: CallbackContext) -> None:
        models_list = ', '.join(self.valid_models)  # Dynamically generate the list of valid models
        help_text = f"""Welcome to the Whisper Transcriber Bot!

Version number: {self.version_number}

How to Use:
- Send any supported media URL to have its audio transcribed.
- Use /model to change the transcription model (currently set to '{self.model}').
- Use /help or /about to display this help message.

Available Models:
{models_list}

Code by FlyingFathead.

Source Code:
https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/

Disclaimer:
The original author is not responsible for how this bot is utilized. All code and outputs are provided 'AS IS' without warranty of any kind. Users assume full responsibility for the operation and output of the bot. Use at your own risk.
"""
        await update.message.reply_text(help_text)

    async def model_command(self, update: Update, context: CallbackContext) -> None:
        # If no specific model is specified, just report the current model
        if not context.args:
            await update.message.reply_text(f"The current transcription model is set to: {self.model}")
        else:
            new_model = context.args[0].strip()
            if new_model in self.valid_models:
                self.model = new_model
                self.config.set('WhisperSettings', 'Model', new_model)
                with open('config/config.ini', 'w') as configfile:
                    self.config.write(configfile)
                await update.message.reply_text(f"Model updated to {new_model}.")
            else:
                models_list = ', '.join(self.valid_models)
                await update.message.reply_text(f"Invalid model specified. Available models: {models_list}.")

    def run(self):
        loop = asyncio.get_event_loop()
        
        for sig in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s, loop)))

        connected = False
        while not connected:
            try:
                self.application = Application.builder().token(self.token).build()

                # Adding handlers
                text_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
                self.application.add_handler(text_handler)

                # Here's where you add the command handler for /help
                help_handler = CommandHandler(['help', 'about'], self.help_command)
                self.application.add_handler(help_handler)

                # Adding model command handler
                model_handler = CommandHandler('model', self.model_command)
                self.application.add_handler(model_handler)

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
    bot = TranscriberBot()
    bot.run()
