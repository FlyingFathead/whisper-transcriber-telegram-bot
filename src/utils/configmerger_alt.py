#!/usr/bin/env python3
"""
configmerger.py
Line-based config merger for INI-like files, preserving comments and ordering.

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
  --no-backup       Do NOT create a backup of the main config before writing 
                    (by default, a backup is created in the same directory).
  
If no changes are found, it does nothing. If new sections or new keys are found,
they’re added. If old keys differ in value, they’re updated, and we print a
summary so you can see exactly what changed.

This script preserves:
  - The main file's existing comments (# or ;) and ordering
  - Avoids duplicating lines if they already exist
  - Merges lines from the custom file into the correct sections or 
    appends them at the end if the section doesn't exist
"""

import sys
import os
import shutil

# ----------------------------------------------------------------
# Global defaults & variables:
# ----------------------------------------------------------------

DEFAULT_MAIN_CONFIG = "config/config.ini"

skip_prompt = False
use_backup = True
main_cfg = None
custom_cfg = None

# ----------------------------------------------------------------
# End of global “DECLARATIONS”
# ----------------------------------------------------------------

def usage():
    print(__doc__.strip())

def make_backup_if_needed(file_path):
    """
    Creates <file_path>.bak (or .bak1, .bak2, ...) in the same directory, 
    if file_path exists. Returns the backup filename or None if no backup.
    """
    if not os.path.isfile(file_path):
        return None

    base, ext = os.path.splitext(file_path)
    backup_candidate = base + ".bak"
    i = 1
    while os.path.exists(backup_candidate):
        backup_candidate = base + f"_{i}.bak"
        i += 1

    shutil.copy2(file_path, backup_candidate)
    return backup_candidate

def parse_ini_lines(file_path):
    """
    Parse an INI-like file line-by-line into a list of dicts. 
    Each dict has:
      {
        "type": "section"/"kv"/"comment"/"blank"/"unknown",
        "raw": original_line,
        "section": section_name if in a section,
        "key": if type="kv",
        "value": if type="kv",
        "name": if type="section"
      }
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    lines_data = []
    current_section = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            stripped = raw.strip()

            if stripped == "":
                # blank
                lines_data.append({
                    "type": "blank",
                    "raw": raw
                })
                continue

            if stripped.startswith("#") or stripped.startswith(";"):
                # comment
                lines_data.append({
                    "type": "comment",
                    "raw": raw
                })
                continue

            if stripped.startswith("[") and stripped.endswith("]"):
                # section
                sec_name = stripped[1:-1].strip()
                current_section = sec_name
                lines_data.append({
                    "type": "section",
                    "name": sec_name,
                    "raw": raw
                })
                continue

            if "=" in stripped:
                # key = value
                key_part, val_part = stripped.split("=", 1)
                key = key_part.strip()
                val = val_part.strip()
                lines_data.append({
                    "type": "kv",
                    "section": current_section,
                    "key": key,
                    "value": val,
                    "raw": raw
                })
                continue

            # unknown line type
            lines_data.append({
                "type": "unknown",
                "raw": raw
            })

    return lines_data

def line_exists_in(lines_list, raw_line):
    """Return True if there's an item in lines_list with item['raw'] == raw_line."""
    return any(item["raw"] == raw_line for item in lines_list)

def find_section_index(lines_list, section_name):
    """Return the index of the line that is [section_name], else -1."""
    for i, item in enumerate(lines_list):
        if item["type"] == "section" and item["name"] == section_name:
            return i
    return -1

def find_kv_in_section(lines_list, section_name, key_name):
    """
    Return (index, item) of the kv line for 'key_name' in [section_name], else (-1, None).
    We scan from the section line downward until next section or end of file.
    """
    sec_idx = find_section_index(lines_list, section_name)
    if sec_idx < 0:
        return -1, None

    i = sec_idx + 1
    while i < len(lines_list):
        if lines_list[i]["type"] == "section":
            break
        if lines_list[i]["type"] == "kv" and lines_list[i]["section"] == section_name:
            if lines_list[i]["key"] == key_name:
                return i, lines_list[i]
        i += 1
    return -1, None

def merge_linebased(main_file, custom_file):
    """
    Merge custom_file -> main_file line by line, preserving the main file’s
    order, comments, blank lines, etc. Return (merged_lines, changes).
    'changes' is a dict summarizing new sections, updated keys, new keys, new comments.
    """
    main_data = parse_ini_lines(main_file)
    custom_data = parse_ini_lines(custom_file)

    changes = {
        "new_sections": [],
        "updated_keys": [],
        "new_keys": [],
        "new_comments": []
    }

    i = 0
    while i < len(custom_data):
        citem = custom_data[i]
        ctype = citem["type"]
        craw = citem["raw"]

        if ctype == "section":
            sec_name = citem["name"]
            sec_idx = find_section_index(main_data, sec_name)
            if sec_idx < 0:
                # new entire section => we copy the section line & subsequent lines
                changes["new_sections"].append(sec_name)
                if not line_exists_in(main_data, craw):
                    main_data.append(citem)

                # Also copy subsequent lines for that section from custom
                j = i + 1
                while j < len(custom_data):
                    if custom_data[j]["type"] == "section":
                        break
                    if not line_exists_in(main_data, custom_data[j]["raw"]):
                        jtype = custom_data[j]["type"]
                        if jtype == "kv":
                            changes["new_keys"].append(
                                (sec_name, custom_data[j]["key"], custom_data[j]["value"])
                            )
                            main_data.append(custom_data[j])
                        elif jtype == "comment":
                            changes["new_comments"].append(custom_data[j]["raw"])
                            main_data.append(custom_data[j])
                        else:
                            # blank, unknown, etc.
                            main_data.append(custom_data[j])
                    j += 1
                i = j
                continue

        elif ctype == "kv":
            sec = citem["section"]
            if sec is None:
                # KV with no section => just append if not dup
                if not line_exists_in(main_data, craw):
                    changes["new_keys"].append(("", citem["key"], citem["value"]))
                    main_data.append(citem)
            else:
                kv_idx, kv_item = find_kv_in_section(main_data, sec, citem["key"])
                if kv_idx < 0:
                    # new key
                    sec_idx = find_section_index(main_data, sec)
                    if sec_idx < 0:
                        # that section doesn't exist at all, or hasn't appeared
                        # Just append a new section line, then kv
                        changes["new_sections"].append(sec)
                        new_section_line = {
                            "type": "section",
                            "name": sec,
                            "raw": f"[{sec}]"
                        }
                        main_data.append(new_section_line)
                        changes["new_keys"].append((sec, citem["key"], citem["value"]))
                        main_data.append(citem)
                    else:
                        # insert in that section
                        # find insertion point
                        insert_pos = sec_idx + 1
                        while insert_pos < len(main_data):
                            if main_data[insert_pos]["type"] == "section":
                                break
                            insert_pos += 1
                        if not line_exists_in(main_data, craw):
                            changes["new_keys"].append((sec, citem["key"], citem["value"]))
                            main_data.insert(insert_pos, citem)
                else:
                    # key exists => update if needed
                    if kv_item["value"] != citem["value"]:
                        changes["updated_keys"].append(
                            (sec, kv_item["key"], kv_item["value"], citem["value"])
                        )
                        kv_item["value"] = citem["value"]
                        kv_item["raw"] = f"{kv_item['key']}={kv_item['value']}"

        elif ctype == "comment":
            if not line_exists_in(main_data, craw):
                changes["new_comments"].append(craw)
                main_data.append(citem)
        elif ctype == "blank":
            if not line_exists_in(main_data, craw):
                main_data.append(citem)
        else:
            # unknown
            if not line_exists_in(main_data, craw):
                main_data.append(citem)

        i += 1

    return main_data, changes

def summarize_changes(changes):
    new_secs = changes["new_sections"]
    upd_keys = changes["updated_keys"]
    new_keys = changes["new_keys"]
    new_cmts = changes["new_comments"]

    if new_secs:
        print(f"\nNew sections to be added: {', '.join(new_secs)}")
    else:
        print("\nNo new sections.")

    if upd_keys:
        print("\nKeys that will be UPDATED (old -> new):")
        for (sec, k, oldv, newv) in upd_keys:
            print(f"  [{sec}] {k}: '{oldv}' -> '{newv}'")
    else:
        print("\nNo existing keys will be updated.")

    if new_keys:
        print("\nKeys that will be ADDED:")
        for (sec, k, val) in new_keys:
            if sec:
                print(f"  [{sec}] {k} = {val}")
            else:
                print(f"  {k} = {val}  # (no section)")
    else:
        print("No new keys will be added.")

    if new_cmts:
        print("\nComment lines that will be ADDED (unique):")
        for c in new_cmts:
            print(f"  {c}")
    else:
        print("No new comment lines will be added.")

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
            if main_cfg is None:
                main_cfg = arg
            elif custom_cfg is None:
                custom_cfg = arg
            else:
                print(f"[ERROR] Unexpected extra argument: {arg}")
                usage()
                sys.exit(1)
            i += 1

    # If user only provided 1 file => treat it as custom, and use DEFAULT_MAIN_CONFIG
    if main_cfg and not custom_cfg:
        custom_cfg = main_cfg
        main_cfg = DEFAULT_MAIN_CONFIG

    if not main_cfg or not custom_cfg:
        print("\n[ERROR] Not enough .ini file arguments provided.")
        usage()
        sys.exit(1)

    if not os.path.isfile(main_cfg):
        print(f"[ERROR] Main config file '{main_cfg}' not found.")
        sys.exit(1)
    if not os.path.isfile(custom_cfg):
        print(f"[ERROR] Custom config file '{custom_cfg}' not found.")
        sys.exit(1)

    print(f"--- Merging '{custom_cfg}' into '{main_cfg}' (line-based) ---")

    merged_data, changes = merge_linebased(main_cfg, custom_cfg)
    total_changes = (
        len(changes["new_sections"]) +
        len(changes["updated_keys"]) +
        len(changes["new_keys"]) +
        len(changes["new_comments"])
    )
    if total_changes == 0:
        print("\nNo changes found. Exiting without modifying.")
        return

    summarize_changes(changes)

    # Confirm
    if not skip_prompt:
        choice = input("\nProceed with merging these changes? (y/N) ").strip().lower()
        if choice != 'y':
            print("Aborting merge. No changes were written.")
            return

    # Backup if needed
    backup_file = None
    if use_backup:
        backup_file = make_backup_if_needed(main_cfg)
        if backup_file:
            print(f"[Backup] Created backup: {backup_file}")

    # Write final lines
    with open(main_cfg, "w", encoding="utf-8") as f:
        for item in merged_data:
            f.write(item["raw"] + "\n")

    print(f"\nDone merging. Changes have been written to: {main_cfg}")
    if backup_file:
        print(f"If needed, you can restore the old version from: {backup_file}")

if __name__ == "__main__":
    main()
