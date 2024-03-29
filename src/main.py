# main.py
# ~~~~~~~
# openai-whisper transcriber-bot for Telegram

# version of this program
version_number = "0.07.2"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import re
import signal
import asyncio
import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CallbackContext

# Adjust import paths based on new structure
from transcription_handler import process_url_message
from utils.bot_token import get_bot_token
from utils.language_selection import ask_language
from utils.utils import print_startup_message

# Configure basic logging
logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Call the startup message function
print_startup_message(version_number)

class TranscriberBot:

    # version of this program
    version_number = version_number

    def __init__(self):
        self.token = get_bot_token()
        self.task_queue = asyncio.Queue() # queue tasks        
        self.is_processing = asyncio.Lock()  # Lock to ensure one transcription at a time

    async def handle_message(self, update: Update, context: CallbackContext) -> None:
        logger.info("Received a message.")
        if update.message and update.message.text:
            urls = re.findall(r'(https?://\S+)', update.message.text)

            if urls:
                await self.task_queue.put((update.message.text, context.bot, update))
                # Immediately capture the queue length after adding a job.
                queue_length = self.task_queue.qsize()

                logger.info(f"Task added to the queue. Current queue size: {queue_length}")

                # Adjust messaging logic based on queue size and processing status.
                # If the queue length is 1 and no task is currently processing, then the job is next.
                # Otherwise, if the queue has more items or a task is processing, provide accurate positioning.
                if queue_length == 1 and not self.is_processing.locked():
                    await update.message.reply_text(
                        "Your request has been added to the queue and will be processed next."
                    )
                else:
                    # Informs user accurately about their position in the queue.
                    await update.message.reply_text(
                        f"Your request has been added to the queue. There are {queue_length - 1} jobs ahead of yours."
                    )
            else:
                await update.message.reply_text(
                    "No valid URL detected in your message. Please send a message that includes a valid URL."
                )

    async def process_queue(self):
        while True:
            message_text, bot, update = await self.task_queue.get()
            async with self.is_processing:  # Ensure one job at a time
                await process_url_message(message_text, bot, update)
            self.task_queue.task_done()

    async def shutdown(signal, loop):
        """Cleanup tasks tied to the service's shutdown."""
        print(f"Received exit signal {signal.name}...")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

        [task.cancel() for task in tasks]

        print(f"Cancelling {len(tasks)} outstanding tasks")
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    def run(self):

        loop = asyncio.get_event_loop()        

        # Using 'sig' as the variable name to avoid conflict with the 'signal' module
        for sig in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s, loop)))

        self.application = Application.builder().token(self.token).build()
        text_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        self.application.add_handler(text_handler)

        loop.create_task(self.process_queue())
        self.application.run_polling()

if __name__ == '__main__':
    bot = TranscriberBot()
    bot.run()