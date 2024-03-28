# main.py
# ~~~~~~~
# openai-whisper transcriber-bot for Telegram
# v0.03
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import asyncio
import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CallbackContext

# Adjust import paths based on new structure
from transcription_handler import process_url_message
from utils.bot_token import get_bot_token  # Updated path

# Configure basic logging
logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeURLBot:
    def __init__(self):
        self.token = get_bot_token()
        self.is_processing = asyncio.Lock()  # Lock to ensure one transcription at a time

    async def handle_message(self, update: Update, context: CallbackContext) -> None:
        if update.message and update.message.text:
            # Attempt to acquire the lock
            if self.is_processing.locked():
                await update.message.reply_text("Please wait, another transcription is currently being processed.")
                return

            async with self.is_processing:  # Lock is acquired here
                await process_url_message(update.message.text, context.bot, update)
                # Lock is automatically released here

    def run(self):
        application = Application.builder().token(self.token).build()
        text_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        application.add_handler(text_handler)
        application.run_polling()

if __name__ == '__main__':
    bot = YouTubeURLBot()
    bot.run()