#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# reencode_to_target_size.py
#
#   Re-encode audio/video to MP3, ensuring file size is under some target MB.
#   - Binary search on bitrate from 8 kbps → 320 kbps
#   - Up to max_iterations
#   - If we can’t improve or we’re within a tolerance, we stop early.
#
# (c) 2024. Use at your own risk.

import sys
import os
import subprocess
import shutil

# =========== GLOBALS ============
TELEGRAM_BOT_SIZE_LIMIT_MB = 20.0
SAFETY_MARGIN_MB           = 0.1
DEFAULT_TARGET_MB          = TELEGRAM_BOT_SIZE_LIMIT_MB - SAFETY_MARGIN_MB

DEFAULT_MAX_ITERATIONS     = 12  # Increase default to e.g. 12
STALE_TRIES_LIMIT          = 2   # If we fail to improve for 2 consecutive tries, bail out

# TOLERANCE: If we’re within X bytes of the target, we’ll consider that “close enough”
SIZE_TOLERANCE_BYTES = 200_000  # ~200 KB; set to 0 to disable
# =================================

def hz_line(char='-'):
    import shutil
    try:
        cols, _ = shutil.get_terminal_size(fallback=(80, 24))
    except:
        cols = 80
    print(char * cols)

def get_duration_seconds(inputfile):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        inputfile
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration_str = proc.stdout.strip()
        return float(duration_str) if duration_str else None
    except Exception as e:
        print(f"[ERROR] ffprobe failed: {e}")
        return None

def reencode_mp3_strict(inputfile, target_mb, max_iterations):
    duration = get_duration_seconds(inputfile)
    if not duration:
        print("[ERROR] Could not get duration. Aborting.")
        return None

    target_bytes = int(target_mb * 1024 * 1024)
    min_bitrate = 8
    max_bitrate = 320

    base, ext = os.path.splitext(inputfile)
    ext_lower = ext.lower()
    if ext_lower not in (".mp3", ".aac", ".wav", ".m4a", ".mp4", ".ogg", ".flac"):
        ext_lower = ".mp3"

    outputfile = f"{base}_{target_mb}MB_recode.mp3"

    hz_line()
    print(f"[INFO] Input:           {inputfile}")
    print(f"[INFO] Duration:        {duration:.1f} s")
    print(f"[INFO] Target:          {target_mb} MB → {target_bytes} bytes")
    print(f"[INFO] Output file:     {outputfile}")
    print(f"[INFO] Max iterations:  {max_iterations}")
    hz_line()

    best_size = 0
    best_bitrate = None
    stale_tries = 0     # how many times in a row we fail to improve
    prev_size = None    # to see if two consecutive attempts produce same size

    iteration = 1
    while iteration <= max_iterations and min_bitrate <= max_bitrate:
        current_bitrate = (min_bitrate + max_bitrate) // 2
        print(f"[Iteration {iteration}] Trying bitrate: {current_bitrate} kbps")

        tmp_out = outputfile + ".temp"
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-i", inputfile,
            "-vn",
            "-c:a", "libmp3lame",
            f"-b:a", f"{current_bitrate}k",
            "-f", "mp3",
            tmp_out
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] ffmpeg encoding failed: {e}")
            return None

        actual_size = os.path.getsize(tmp_out)
        print(f"  -> size: {actual_size} bytes")

        # If under the limit
        if actual_size <= target_bytes:
            # Check if it’s bigger than our previous best
            if actual_size > best_size:
                best_size = actual_size
                best_bitrate = current_bitrate
                shutil.copyfile(tmp_out, outputfile + ".best")
                print(f"  -> New best: {best_size} bytes at {best_bitrate} kbps")
                stale_tries = 0  # reset
            else:
                # Didn’t improve
                stale_tries += 1
                print(f"  -> Not an improvement. stale_tries={stale_tries}")
            # Then try increasing min_bitrate to see if we can get closer
            min_bitrate = current_bitrate + 1
        else:
            # Over the limit => lower bitrate
            stale_tries += 1
            print(f"  -> Over target. stale_tries={stale_tries}")
            max_bitrate = current_bitrate - 1

        # If consecutive attempts produce same size, we might be pinned by encoder
        if prev_size is not None and prev_size == actual_size:
            stale_tries += 1
            print("  -> Same file size as previous iteration. Possibly pinned by encoder constraints.")

        prev_size = actual_size
        os.remove(tmp_out)

        # Check if we’re within tolerance
        if best_size > 0 and SIZE_TOLERANCE_BYTES > 0:
            diff = target_bytes - best_size
            if 0 <= diff <= SIZE_TOLERANCE_BYTES:
                print(f"[INFO] Within {SIZE_TOLERANCE_BYTES} bytes of target; stopping early.")
                break

        # If we’ve not improved for too many tries, break
        if stale_tries >= STALE_TRIES_LIMIT:
            print(f"[INFO] Reached stale tries limit ({STALE_TRIES_LIMIT}). Stopping.")
            break

        iteration += 1
        print()

    best_file = outputfile + ".best"
    if os.path.exists(best_file):
        # Move final
        shutil.move(best_file, outputfile)
        final_size = os.path.getsize(outputfile)
        print(f"[FINAL] Best bitrate: {best_bitrate} kbps, size: {final_size} bytes")
        if final_size > target_bytes:
            print("[WARN] Final is STILL OVER target, but we can't do better with given constraints.")
        else:
            print("[INFO] Final is under target!")
        return outputfile
    else:
        print("[ERROR] Could not get under target (or no .best file).")
        return None

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <inputfile> [<targetMB>] [<max_iterations>]")
        sys.exit(1)

    inputfile = sys.argv[1]
    if not os.path.isfile(inputfile):
        print(f"[ERROR] File not found: {inputfile}")
        sys.exit(2)

    if len(sys.argv) >= 3:
        target_mb = float(sys.argv[2])
    else:
        target_mb = DEFAULT_TARGET_MB

    if len(sys.argv) >= 4:
        max_iterations = int(sys.argv[3])
    else:
        max_iterations = DEFAULT_MAX_ITERATIONS

    result = reencode_mp3_strict(inputfile, target_mb, max_iterations)
    if result:
        print(f"[DONE] Re-encoded: {result}")
    else:
        print("[FAIL] No acceptable file produced.")

if __name__ == "__main__":
    main()
