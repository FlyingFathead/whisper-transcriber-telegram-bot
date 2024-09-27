# utils.py

import os
import re
import shutil
import sys
import datetime
import asyncio

# set `now`
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# print term width horizontal line
def hz_line(character='-'):
    terminal_width = shutil.get_terminal_size().columns
    line = character * terminal_width
    print(line)
    sys.stdout.flush()  # Flush the output to the terminal immediately

# print the startup message
def print_startup_message(version_number):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hz_line()
    print(f"[{now}] Telegram video transcriber bot v.{version_number} is starting up...", flush=True)
    hz_line()

# safe splitting
def safe_split_message(message, max_length=4000):
    """
    Safely split a message into chunks without breaking words or HTML tags.
    """
    parts = []
    while len(message) > max_length:
        # Try to split at the last newline character before max_length
        split_pos = message.rfind('\n', 0, max_length)
        # If no newline character, split at the last space
        if split_pos == -1:
            split_pos = message.rfind(' ', 0, max_length)
        # If no space, split at max_length
        if split_pos == -1:
            split_pos = max_length
        # Add the chunk to parts
        parts.append(message[:split_pos])
        # Remove the chunk from message
        message = message[split_pos:].lstrip()
    parts.append(message)
    return parts

# def safe_split_message(content, max_length=3996):
#     """
#     Splits the content into chunks that are less than max_length.
#     Tries to split at spaces or newlines for readability.
#     """
#     import re
#     words = re.findall(r'\S+\s*', content)
#     chunks = []
#     current_chunk = ''

#     for word in words:
#         if len(current_chunk) + len(word) <= max_length:
#             current_chunk += word
#         else:
#             if current_chunk:
#                 chunks.append(current_chunk)
#             if len(word) > max_length:
#                 # Split the long word
#                 for i in range(0, len(word), max_length):
#                     chunks.append(word[i:i+max_length])
#                 current_chunk = ''
#             else:
#                 current_chunk = word

#     if current_chunk:
#         chunks.append(current_chunk)
#     return chunks
