# resemblyzer_safety_check.py
#
# This script checks for Resemblyzer's unsafe loading method (present in v.0.1.4) and secures it.
# It's done by setting `weights_only=True` in Resemblyzer's `voice_encoder.py` if it's not set.
# The script creates a backup of the original file before making any changes.
#
# To undo, you can `pip uninstall resemblyzer` and reinstall it with `pip install resemblyzer`
#
# === How to run ===
# After installing `resemblyzer` via `pip install resemblyzer`, you can run this script.
# NOTE: This has _only_ been tested on `resemblyzer-0.1.4`. Later versions might introduce
# different code altogether, and you should not run this if you don't know what you're doing!
#
# To install v0.1.4 specifically, use: `pip install resemblyzer==0.1.4`
#
# (c) 2024 FlyingFathead (https://github.com/FlyingFathead)
# (From: https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/)

import os
import re
import logging
import resemblyzer
import shutil
from datetime import datetime
import pkg_resources

# Supported Resemblyzer version
SUPPORTED_RESEMBLYZER_VERSION = "0.1.4"

def find_voice_encoder_py():
    """
    Locate the voice_encoder.py file within the installed Resemblyzer package.
    """
    resemblyzer_dir = os.path.dirname(resemblyzer.__file__)
    voice_encoder_path = os.path.join(resemblyzer_dir, 'voice_encoder.py')
    return voice_encoder_path

def backup_file(file_path):
    """
    Create a backup of the specified file.
    """
    backup_path = file_path + '.bak'
    if os.path.exists(backup_path):
        # If a backup already exists, create a new one with a timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        backup_path = file_path + f'.bak_{timestamp}'
    shutil.copy2(file_path, backup_path)
    logging.info(f"Created a backup of the original file at {backup_path}")
    return backup_path

def check_and_modify_voice_encoder(voice_encoder_path):
    """
    Check if torch.load in voice_encoder.py includes weights_only=True.
    If not, modify the code to include it, preserving indentation.
    """
    with open(voice_encoder_path, 'r') as f:
        lines = f.readlines()

    modified = False
    for i, line in enumerate(lines):
        if 'torch.load(' in line:
            # Check if weights_only is already present
            if 'weights_only' not in line:
                logging.info("Found torch.load without weights_only=True.")
                # Capture indentation
                indentation_match = re.match(r'^(\s*)', line)
                indentation = indentation_match.group(1) if indentation_match else ''
                # Preserve any code before 'torch.load(' (e.g., 'checkpoint = ')
                line_content = line.strip()
                line_before_load = line_content.split('torch.load(')[0]
                # Modify the line to include weights_only=True
                # Use regex to safely add the parameter
                pattern = r'(torch\.load\()([^\)]*)(\))'
                match = re.search(pattern, line_content)
                if match:
                    before_args = match.group(1)
                    args = match.group(2)
                    after_args = match.group(3)
                    if args.strip():
                        new_args = args + ', weights_only=True'
                    else:
                        new_args = 'weights_only=True'
                    # Reconstruct the line with preserved indentation
                    new_line = indentation + line_before_load + before_args + new_args + after_args + '\n'
                    lines[i] = new_line
                    modified = True
                    logging.info("Modified torch.load call to include weights_only=True.")
                else:
                    logging.warning("Could not parse torch.load line for modification.")
            else:
                logging.info("weights_only argument already present in torch.load call.")
    if modified:
        # Create a backup before writing changes
        backup_path = backup_file(voice_encoder_path)
        # Write back the modified contents
        with open(voice_encoder_path, 'w') as f:
            f.writelines(lines)
        logging.info(f"Modified {voice_encoder_path} to include weights_only=True in torch.load.")
        logging.info(f"The original file has been backed up at {backup_path}.")
    else:
        logging.info(f"No modifications made to {voice_encoder_path}.")

def main():
    logging.basicConfig(level=logging.INFO)
    # Get installed Resemblyzer version
    try:
        installed_version = pkg_resources.get_distribution("Resemblyzer").version
    except pkg_resources.DistributionNotFound:
        logging.error("Resemblyzer is not installed.")
        return
    logging.info(f"Installed Resemblyzer version: {installed_version}")

    if installed_version != SUPPORTED_RESEMBLYZER_VERSION:
        logging.warning(f"This script is designed for Resemblyzer version {SUPPORTED_RESEMBLYZER_VERSION}, but version {installed_version} is installed.")
        proceed = input("Do you want to proceed with checking and modifying the code? (yes/no): ")
        if proceed.lower() != 'yes':
            logging.info("Exiting without making any changes.")
            return
        else:
            logging.info("Proceeding with checking and modifying the code.")

    voice_encoder_path = find_voice_encoder_py()
    if os.path.exists(voice_encoder_path):
        logging.info(f"Found voice_encoder.py at {voice_encoder_path}")
        check_and_modify_voice_encoder(voice_encoder_path)
    else:
        logging.error("voice_encoder.py not found in the Resemblyzer package.")

if __name__ == "__main__":
    main()
