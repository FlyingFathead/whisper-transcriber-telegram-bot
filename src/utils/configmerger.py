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
  -y, --yes      Merge without prompting (skip confirmation).
  -h, --help     Print this help and exit.

If no changes are found, it does nothing. If new sections or new keys are found,
they’re added.  If old keys differ in value, they’re updated, and we print a
summary so you can see exactly what changed.
"""

import sys
import os
import configparser

DEFAULT_MAIN_CONFIG = "config/config.ini"

def usage():
    print(__doc__.strip())

def merge_inis(main_path, custom_path, skip_prompt=False):
    # Load both files into configparser objects
    main_parser = configparser.ConfigParser()
    custom_parser = configparser.ConfigParser()

    # configparser normally lowercases section names by default. 
    # If you need case sensitivity, set: main_parser.optionxform=str
    main_parser.read(main_path, encoding='utf-8')
    custom_parser.read(custom_path, encoding='utf-8')

    # We'll track changes in dict form:
    # updated_keys = [(section, key, old_value, new_value)]
    # new_keys     = [(section, key, new_value)]
    updated_keys = []
    new_keys = []
    new_sections = []

    # 1) Merge: For each section in custom, for each key in that section:
    for section in custom_parser.sections():
        # If main lacks that section, we add it
        if not main_parser.has_section(section):
            new_sections.append(section)
            main_parser.add_section(section)

        # now read key-values
        for key, custom_value in custom_parser[section].items():
            if main_parser.has_option(section, key):
                old_value = main_parser[section][key]
                if old_value != custom_value:
                    updated_keys.append((section, key, old_value, custom_value))
            else:
                new_keys.append((section, key, custom_value))

    # 2) Print summary
    print(f"\n--- Merging '{custom_path}' into '{main_path}' ---")

    if new_sections:
        print(f"\nNew sections to be added: {', '.join(new_sections)}")

    if updated_keys:
        print("\nKeys that will be UPDATED (old -> new):")
        for (s, k, oldv, newv) in updated_keys:
            print(f"  [{s}] {k}: '{oldv}' -> '{newv}'")
    else:
        print("\nNo existing keys will be updated.")

    if new_keys:
        print("\nKeys that will be ADDED:")
        for (s, k, newv) in new_keys:
            print(f"  [{s}] {k} = {newv}")
    else:
        print("No new keys will be added.")

    # If absolutely nothing changes:
    if not new_sections and not updated_keys and not new_keys:
        print("\nNo changes found. Exiting without modifying.")
        return

    # 3) Prompt user unless skip_prompt
    if not skip_prompt:
        choice = input("\nProceed with merging these changes? (y/N) ").strip().lower()
        if choice != 'y':
            print("Aborting. No changes written.")
            return

    # 4) Actually perform changes in memory
    # - For each updated key, write new_value
    # - For each new key, also write
    # - For each new section, that’s already added (above), so just do the keys.
    for (s, k, oldv, newv) in updated_keys:
        main_parser[s][k] = newv

    for (s, k, newv) in new_keys:
        main_parser[s][k] = newv

    # 5) Write out final result
    with open(main_path, 'w', encoding='utf-8') as f:
        main_parser.write(f)

    print(f"\nDone. Changes have been written to: {main_path}")

def main():
    args = sys.argv[1:]
    if not args:
        usage()
        sys.exit(1)

    skip_prompt = False
    main_cfg = None
    custom_cfg = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif arg in ("-y", "--yes"):
            skip_prompt = True
            i += 1
        else:
            if main_cfg is None:
                main_cfg = arg
            elif custom_cfg is None:
                custom_cfg = arg
            else:
                print(f"[ERROR] Unexpected extra argument: {arg}")
                sys.exit(1)
            i += 1

    # If only 1 file given => treat it as custom,
    # and use the default main config
    if main_cfg and not custom_cfg:
        custom_cfg = main_cfg
        main_cfg = DEFAULT_MAIN_CONFIG

    if not main_cfg or not custom_cfg:
        usage()
        print("\n[ERROR] Not enough config-file arguments provided.")
        sys.exit(1)

    # Check existence of main_cfg
    if not os.path.isfile(main_cfg):
        print(f"[ERROR] Main config file '{main_cfg}' not found.")
        sys.exit(1)
    # Check existence of custom_cfg
    if not os.path.isfile(custom_cfg):
        print(f"[ERROR] Custom config file '{custom_cfg}' not found.")
        sys.exit(1)

    merge_inis(main_cfg, custom_cfg, skip_prompt=skip_prompt)

if __name__ == "__main__":
    main()