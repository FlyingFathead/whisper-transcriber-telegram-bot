#!/usr/bin/env python3

# configmerger.py (config.ini merger) for the Whisper Transcriber Telegram Bot
# (https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/)

"""
configmerger.py
Merge settings from a "custom" .INI file into a "main" .INI file using Python's configparser.

Usage:
  configmerger.py [OPTIONS] <custom_config.ini>
  configmerger.py [OPTIONS] <main_config.ini> <custom_config.ini>

Examples:
  1) Merge 'alt_config.ini' into the default main config 'config/config.ini':
     ./configmerger.py config/alt_config.ini

  2) Merge 'my_custom.ini' into 'some_main.ini' (explicit main):
     ./configmerger.py some_main.ini my_custom.ini

  3) Same as #2, but skip user confirmation:
     ./configmerger.py some_main.ini my_custom.ini --yes

Options:
  -y, --yes         Merge without prompting (skip confirmation).
  -h, --help        Print this help and exit.
  --no-backup       Do NOT create a backup of the main config before writing.
                    (By default, a backup is created.)
  
If no changes are found, it does nothing. If new sections or new keys are found,
they’re added. If old keys differ in value, they’re updated, and we print a
summary so you can see exactly what changed.
"""

import sys
import os
import shutil
import configparser
from datetime import datetime

# ----------------------------------------------------------------
# ALL VARIABLES DECLARED HERE
# ----------------------------------------------------------------

DEFAULT_MAIN_CONFIG = "config/config.ini"
DEFAULT_BACKUP_LOCATION = "config/"
skip_prompt = False
use_backup = True
main_cfg = None
custom_cfg = None

# ----------------------------------------------------------------
# END OF GLOBAL “DECLARATIONS”
# ----------------------------------------------------------------

def usage():
    print(__doc__.strip())

def make_backup_if_needed(main_path, backup_dir):
    """
    Creates a backup of 'main_path' in backup_dir, if main_path exists.
    
    We name it something like:
        backup_dir / <basename_of_main_cfg>.bak
        or if that exists, .bak1, .bak2, ...
    
    Returns the backup file path, or None if no backup was made.
    """
    if not os.path.isfile(main_path):
        # main_path doesn't exist -> no backup
        return None

    # Make sure backup_dir exists or create it
    if not os.path.exists(backup_dir):
        print(f"[INFO] Backup directory '{backup_dir}' does not exist; attempting to create.")
        try:
            os.makedirs(backup_dir, exist_ok=True)
        except Exception as e:
            print(f"[ERROR] Could not create backup directory '{backup_dir}': {e}")
            sys.exit(1)

    # We'll store backups with name like "config.ini.bak", "config.ini.bak1", ...
    base_name = os.path.basename(main_path)  # e.g. "config.ini"
    name_no_ext, ext = os.path.splitext(base_name)    # ("config", ".ini")
    # Start with "config.ini.bak" inside backup_dir
    backup_candidate = os.path.join(backup_dir, f"{base_name}.bak")

    i = 1
    while os.path.exists(backup_candidate):
        # If e.g. "config.ini.bak" exists, try "config.ini.bak1", etc.
        backup_candidate = os.path.join(backup_dir, f"{base_name}.bak{i}")
        i += 1

    # Copy main_path -> backup_candidate
    shutil.copy2(main_path, backup_candidate)
    return backup_candidate

def merge_inis(main_path, custom_path, skip_prompt=False, use_backup=True):
    """
    Merge the 'custom_path' config into 'main_path' config, preserving sections.

    - If 'use_backup' is True, creates a backup of main_path in DEFAULT_BACKUP_LOCATION before writing changes.
    - If 'skip_prompt' is False, shows a summary and asks user to confirm merging.
    """
    main_parser = configparser.ConfigParser()
    custom_parser = configparser.ConfigParser()

    # Load the INI files
    main_parser.read(main_path, encoding='utf-8')
    custom_parser.read(custom_path, encoding='utf-8')

    # Track changes for user summary
    updated_keys = []   # [(section, key, old_val, new_val)]
    new_keys = []       # [(section, key, new_val)]
    new_sections = []   # [section_name]

    # For each section/key in the custom file:
    for section in custom_parser.sections():
        if not main_parser.has_section(section):
            # Entire new section
            new_sections.append(section)
            main_parser.add_section(section)

        for key, custom_val in custom_parser[section].items():
            if main_parser.has_option(section, key):
                old_val = main_parser[section][key]
                if old_val != custom_val:
                    updated_keys.append((section, key, old_val, custom_val))
            else:
                new_keys.append((section, key, custom_val))

    # Summarize
    print(f"\n--- Merging '{custom_path}' into '{main_path}' ---")

    if new_sections:
        print(f"\nNew sections to be added: {', '.join(new_sections)}")

    if updated_keys:
        print("\nKeys that will be UPDATED (old -> new):")
        for (sec, k, oldv, newv) in updated_keys:
            print(f"  [{sec}] {k}: '{oldv}' -> '{newv}'")
    else:
        print("\nNo existing keys will be updated.")

    if new_keys:
        print("\nKeys that will be ADDED:")
        for (sec, k, newv) in new_keys:
            print(f"  [{sec}] {k} = {newv}")
    else:
        print("No new keys will be added.")

    no_changes = (not new_sections) and (not updated_keys) and (not new_keys)
    if no_changes:
        print("\nNo changes found. Exiting without modifying.")
        return

    # Confirm unless skip_prompt
    if not skip_prompt:
        choice = input("\nProceed with merging these changes? (y/N) ").strip().lower()
        if choice != 'y':
            print("Aborting. No changes written.")
            return

    # If using backups, attempt to back up the main file in DEFAULT_BACKUP_LOCATION
    backup_file = None
    if use_backup:
        backup_file = make_backup_if_needed(main_path, DEFAULT_BACKUP_LOCATION)
        if backup_file:
            print(f"[Backup] Created backup: {backup_file}")
        else:
            # If main_path didn't exist or we couldn't back it up, we just proceed anyway
            pass

    # Perform the merges in memory
    for (sec, k, oldv, newv) in updated_keys:
        main_parser[sec][k] = newv
    for (sec, k, newv) in new_keys:
        main_parser[sec][k] = newv

    # Write out
    with open(main_path, 'w', encoding='utf-8') as f:
        main_parser.write(f)

    print(f"\nDone. Changes have been written to: {main_path}")
    if backup_file:
        print(f"If needed, you can restore the old version from backup: {backup_file}")

def main():
    global skip_prompt, use_backup, main_cfg, custom_cfg

    args = sys.argv[1:]
    if not args:
        usage()
        sys.exit(1)

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif arg in ("-y", "--yes"):
            skip_prompt = True
            i += 1
        elif arg == "--no-backup":
            use_backup = False
            i += 1
        else:
            # This is presumably a config file
            if main_cfg is None:
                main_cfg = arg
            elif custom_cfg is None:
                custom_cfg = arg
            else:
                print(f"[ERROR] Unexpected extra argument: {arg}")
                sys.exit(1)
            i += 1

    # If only one file was given => treat it as "custom" and use the default main config
    if main_cfg and not custom_cfg:
        custom_cfg = main_cfg
        main_cfg = DEFAULT_MAIN_CONFIG

    # Check that we indeed have two .ini paths
    if not main_cfg or not custom_cfg:
        print("\n[ERROR] Not enough .ini file arguments provided.")
        usage()
        sys.exit(1)

    # Validate that both exist
    if not os.path.isfile(main_cfg):
        print(f"[ERROR] Main config file '{main_cfg}' not found.")
        sys.exit(1)
    if not os.path.isfile(custom_cfg):
        print(f"[ERROR] Custom config file '{custom_cfg}' not found.")
        sys.exit(1)

    # Now call the actual merger
    merge_inis(
        main_path=main_cfg,
        custom_path=custom_cfg,
        skip_prompt=skip_prompt,
        use_backup=use_backup
    )

if __name__ == "__main__":
    main()
