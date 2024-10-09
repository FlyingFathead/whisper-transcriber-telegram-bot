# utils.py

import os
import re
import html
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

# safe splitting method
def safe_split_message(message, max_length=3500):
    """
    Safely split a message into chunks that do not exceed max_length.
    Handles cases where there are no spaces or suitable separators.
    Ensures that HTML entities and tags are not broken.
    """
    parts = []
    index = 0
    while index < len(message):
        end_index = index + max_length
        if end_index >= len(message):
            # Remaining message fits within max_length
            parts.append(message[index:])
            break
        else:
            # Initialize split position
            split_pos = end_index
            # Adjust split position to avoid breaking HTML entities or tags
            while True:
                # Avoid splitting in the middle of an HTML entity
                if message[split_pos - 1] == '&':
                    split_pos -= 1
                # Avoid splitting in the middle of an HTML tag
                elif message[split_pos - 1] == '<':
                    split_pos -= 1
                else:
                    break
                # Prevent infinite loop
                if split_pos <= index:
                    split_pos = end_index
                    break
            # If we moved the split position backward, check for space
            last_space = message.rfind(' ', index, split_pos)
            if last_space > index:
                split_pos = last_space
            part = message[index:split_pos]
            parts.append(part)
            index = split_pos
    return parts
