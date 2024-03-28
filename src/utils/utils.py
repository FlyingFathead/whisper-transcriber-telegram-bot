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
