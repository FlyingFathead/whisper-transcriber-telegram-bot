# diarize_nemo.py
__version__ = "1.0.1"

import textwrap

SCRIPT_BANNER = textwrap.dedent(f"""
=================================================================================================
diarize_nemo.py v{__version__}

*** NOTE: THIS IS A TEST TEMPLATE; DO NOT RUN THIS UNLESS YOU'RE READY TO FIX IT YOURSELF. ***
*** NO ISSUES WILL BE TRACKED NOR FIXED REGARDING THIS SCRIPT, UNLESS YOU PROVIDE THE SOLUTIONS ***

Created by FlyingFathead and the ChaosWhisperers ...
GitHub repo: https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/

A configurable and hardened script for speaker diarization and transcription using NVIDIA NeMo.
This version balances security and convenience with automatic, verified model downloading and
on-the-fly audio conversion to the required 16kHz mono format.

=================================================================================================
*** IMPORTANT SECURITY NOTE: ***

NeMo's core code (v2.4.0+) is patched against its known CVEs. However, the default PyPI
installation can pull in vulnerable dependencies (e.g., an outdated version of 'transformers').

ALWAYS install with a known-secure dependency set. For the full feature set, use:
  pip install "nemo_toolkit[asr]>=2.4.0" "transformers>=4.53.0" pydub tqdm "torch>=2.0" packaging

In some cases, you might need to install one by one, i.e.:
  pip install -U nemo_toolkit[asr] && pip install -U transformers
  
  (this is to make sure that your transformers library isn't rolling back too much due to NeMo)
  
=================================================================================================
""").strip()

import os
import sys
import logging
import json
import shutil
import argparse
import re
import zipfile
import hashlib
import urllib.request
import urllib.error
from urllib.parse import urlparse
import time
import shlex
import datetime
import platform
import random
import secrets
import signal
import glob
from importlib import metadata as _im

# increase NeMo unzip size
os.environ["NEMO_MAX_UNZIP_BYTES"] = str(8 * 1024 * 1024 * 1024)  # 8 GiB

EXIT_CODE_SUCCESS = 0
EXIT_CODE_GENERIC_ERROR = 1
EXIT_CODE_CLI_ERROR = 2
EXIT_CODE_INTEGRITY_ERROR = 3
EXIT_CODE_PIPELINE_ERROR = 4
EXIT_CODE_CTRL_C = 130

# --- Graceful termination (SIGINT/SIGTERM) ------------------------------------
_SIG_CAUGHT = False

print("[NOTE/WARNING] This script is EXPERIMENTAL. Do NOT leave bug reports on it unless you're willing to fix them yourself.")

def _graceful_exit(signum, frame):
    # Map common signals to human-friendly names (best-effort, portable)
    name = {getattr(signal, "SIGINT", 2): "SIGINT",
            getattr(signal, "SIGTERM", 15): "SIGTERM"}.get(signum, f"signal {signum}")
    global _SIG_CAUGHT
    if not _SIG_CAUGHT:
        _SIG_CAUGHT = True
        try:
            logging.warning(f"\nReceived {name}. Shutting down cleanly...")
        except Exception:
            print(f"\nReceived {name}. Shutting down cleanly...", file=sys.stderr)
    code = EXIT_CODE_CTRL_C if signum == getattr(signal, "SIGINT", 2) else 143
    sys.exit(code)

# error handling/display
class DownloadError(RuntimeError):
    """Raised when a model/artifact download fails in a recoverable, user-facing way."""
    pass

def _install_signal_handlers():
    # Only main thread can set signal handlers
    if hasattr(signal, "signal"):
        for sig_name in ("SIGINT", "SIGTERM"):
            if hasattr(signal, sig_name):
                try:
                    signal.signal(getattr(signal, sig_name), _graceful_exit)
                except Exception:
                    pass

# --- Optional, but recommended, dependencies ---
try: from pydub import AudioSegment; HAVE_PYDUB = True
except ImportError: HAVE_PYDUB = False
try: from packaging.version import Version as _V
except ImportError: _V = None

# Note: tqdm is imported lazily to respect the TQDM_DISABLE environment variable.
class TqdmUpToNoOp:
    def __init__(self, *a, **k): pass
    def update_to(self, *a, **k): pass
    def update(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass

# --- Known Models: The Security Checkpoint ---
# Only public models listed here. NO gated MSDD entries.
KNOWN_MODELS = {
    "parakeet-rnnt-1.1b": {
        "url": "https://huggingface.co/nvidia/parakeet-rnnt-1.1b/resolve/main/parakeet-rnnt-1.1b.nemo",
        "sha256": "535896f014953d945b287ac533560e20da8103c6781b152de4645528e2b60738"
    },
    "titanet_large": {
        "url": "https://huggingface.co/nvidia/speakerverification_en_titanet_large/resolve/main/speakerverification_en_titanet_large.nemo",
        "sha256": "e838520693f269e7984f55bc8eb3c2d60ccf246bf4b896d4be9bcabe3e4b0fe3"
    },
    "vad_multilingual_marblenet": {
        "url": "https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0/resolve/main/frame_vad_multilingual_marblenet_v2.0.nemo",
        "sha256": "84bda37e925ac6fd740c2ced55642cb79f94f81348e1fa0db992ca50d4b09706"
    }
}

# --- Helper and Security Functions ---
_BAD_ZIP_ENTRY = re.compile(r"(?:^/)|(?:\.\.)|(?:\\)|(?:^|/)(?:\.ssh/|\.git/)")

def _validate_models_manifest(obj):
    if not isinstance(obj, dict): raise ValueError("Manifest must be a JSON object of {name: {url, sha256}}")
    for name, info in obj.items():
        if not isinstance(info, dict): raise ValueError(f"In manifest, '{name}' must be an object.")
        if "url" not in info or "sha256" not in info: raise ValueError(f"In manifest, '{name}' must include 'url' and 'sha256' keys.")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", info["sha256"]): raise ValueError(f"In manifest, '{name}' has an invalid sha256 value; must be 64 hex characters.")
        u = urlparse(info["url"])
        if u.scheme not in {"https"}:
            raise ValueError(f"In manifest, '{name}' has an invalid URL scheme '{u.scheme}'; only 'https' is allowed.")
        if not u.netloc:
            raise ValueError(f"In manifest, '{name}' has an invalid URL (empty host).")

def sha256sum(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def scan_nemo_archive(path, max_total_bytes):
    try:
        with zipfile.ZipFile(path) as z:
            total_size = 0
            for n in z.namelist():
                if _BAD_ZIP_ENTRY.search(n) or re.match(r"^[A-Za-z]:[/\\]", n, re.IGNORECASE):
                    logging.critical(f"FATAL: Suspicious/absolute path in archive entry: {n}")
                    sys.exit(EXIT_CODE_INTEGRITY_ERROR)
                info = z.getinfo(n)
                total_size += info.file_size
                if total_size > max_total_bytes:
                    mb = max_total_bytes // (1024*1024)
                    logging.critical(
                        f"FATAL: Archive {path} uncompressed size exceeds limit ({mb} MB). "
                        f"Override with env NEMO_MAX_UNZIP_BYTES, e.g. export NEMO_MAX_UNZIP_BYTES=$((3*1024*1024*1024))")
                    sys.exit(EXIT_CODE_INTEGRITY_ERROR)
    except zipfile.BadZipFile:
        logging.critical(f"FATAL: {path} is not a valid .nemo (zip) archive.")
        sys.exit(EXIT_CODE_INTEGRITY_ERROR)

# model downloader
def _download_atomic(url, dst, no_progress=False, timeout=30, retries=3):
    """
    HTTPS-only atomic downloader with:
      - HF token support via HUGGINGFACE_TOKEN / HF_TOKEN env
      - Proper Accept header for HF 'resolve' endpoints
      - Resume (Range) support if .part exists
      - Exponential backoff with jitter
    """
    # Progress bar class
    if no_progress:
        progress_class = TqdmUpToNoOp
    else:
        from tqdm import tqdm  # lazy import

        class TqdmUpTo(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        progress_class = TqdmUpTo

    dst_part = dst + ".part"

    # Base headers
    headers = {"User-Agent": "Mozilla/5.0"}
    # HF token (works for gated repos, rate limits, etc.)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    # If this looks like a HF "resolve" link, ask for the raw file
    if "huggingface.co" in url and "/resolve/" in url:
        headers["Accept"] = "application/octet-stream"

    for attempt in range(retries):
        try:
            # Resume support: if we already have a partial file, request the remainder
            resume_from = 0
            if os.path.exists(dst_part):
                try:
                    resume_from = os.path.getsize(dst_part)
                except OSError:
                    resume_from = 0
            req_headers = dict(headers)
            if resume_from > 0:
                req_headers["Range"] = f"bytes={resume_from}-"

            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                final_url = r.geturl()
                if not final_url.lower().startswith("https://"):
                    logging.critical(
                        f"FATAL: Download from {url} resulted in a non-HTTPS redirect to {final_url}."
                    )
                    sys.exit(EXIT_CODE_INTEGRITY_ERROR)

                # Figure out total size for progress bar. If resuming and server
                # responded with 206, Content-Range has the full size.
                content_length = r.getheader("Content-Length")
                total_size = int(content_length) if content_length else None
                status = getattr(r, "status", None)  # py3.9+
                if status == 206:
                    # Content-Range: bytes start-end/total
                    cr = r.getheader("Content-Range")
                    if cr and "/" in cr:
                        try:
                            total_size = int(cr.split("/")[-1])
                        except Exception:
                            pass

                # Start progress bar. If resuming, set initial progress to resume_from.
                with progress_class(
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1,
                    desc=os.path.basename(dst),
                ) as t:
                    if total_size is not None:
                        t.update_to(resume_from, tsize=total_size)

                    # Open part file in append mode if resuming
                    mode = "ab" if resume_from > 0 else "wb"
                    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
                    with open(dst_part, mode) as f:
                        bytes_read = resume_from
                        while True:
                            chunk = r.read(1 << 20)  # 1 MiB
                            if not chunk:
                                break
                            f.write(chunk)
                            bytes_read += len(chunk)
                            t.update_to(bytes_read, tsize=total_size)

            os.replace(dst_part, dst)
            return  # success

        except urllib.error.HTTPError as e:
            # Helpful messages for auth/gated cases
            if e.code in (401, 403) and "huggingface.co" in url:
                msg = (
                    "Hugging Face returned {code} for {url}.\n"
                    "- If the repo is gated, accept its license in the browser.\n"
                    "- Export a HF token and try again:\n"
                    "    export HUGGINGFACE_TOKEN=hf_xxx\n"
                ).format(code=e.code, url=url)
                if attempt + 1 >= retries:
                    raise DownloadError(msg) from e
                logging.warning(msg.strip())
            elif e.code == 404:
                logging.critical(f"FATAL: Model URL returned HTTP 404: Not Found.\nURL: {url}")
                sys.exit(EXIT_CODE_INTEGRITY_ERROR)
            else:
                if attempt + 1 >= retries:
                    raise DownloadError(f"Download failed for {url} after {retries} attempts") from e

        except urllib.error.URLError as e:
            reason = getattr(e, "reason", "")
            if "CERTIFICATE_VERIFY_FAILED" in str(reason):
                logging.error(
                    "=" * 80
                    + "\nSSL Certificate verification failed. If behind a corporate proxy,\n"
                    "you may need to set REQUESTS_CA_BUNDLE or SSL_CERT_FILE.\n"
                    + "=" * 80
                )
            if attempt + 1 >= retries:
                raise DownloadError(f"Download failed for {url} after {retries} attempts") from e

        except Exception as e:
            if attempt + 1 >= retries:
                raise DownloadError(f"Download failed for {url} after {retries} attempts") from e

        # Clean up partial if server didn't support Range or we need a clean retry
        try:
            if os.path.exists(dst_part) and os.path.getsize(dst_part) == 0:
                os.remove(dst_part)
        except Exception:
            pass

        delay = 1.5 * (2**attempt) + random.uniform(0, 1)
        logging.warning(f"Download attempt {attempt+1}/{retries} failed. Retrying in {delay:.1f}s...")
        time.sleep(delay)

def fetch_and_verify_model(model_name, cache_dir, allow_online=False, no_progress=False):
    max_unzip = int(os.environ.get("NEMO_MAX_UNZIP_BYTES", 2 * 1024 * 1024 * 1024))
    if os.path.isfile(model_name):
        logging.info(f"[VERIFY] Using local model file: {model_name}")
        scan_nemo_archive(model_name, max_unzip)
        return model_name, sha256sum(model_name)
    if model_name not in KNOWN_MODELS:
        logging.critical(f"FATAL: Unknown model identifier '{model_name}'.")
        sys.exit(EXIT_CODE_CLI_ERROR)
    if not allow_online:
        logging.critical(f"FATAL: Got model name '{model_name}' but network access is disabled (--allow-online=False).")
        sys.exit(EXIT_CODE_CLI_ERROR)
    model_info = KNOWN_MODELS[model_name]
    expected_sha256 = model_info["sha256"]
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.join(cache_dir, os.path.basename(model_info["url"]))

    if not os.path.exists(filename):
        logging.info(f"[DOWNLOAD] Fetching {model_name} from {model_info['url']}")
        try:
            _download_atomic(model_info["url"], filename, no_progress=no_progress)
        except DownloadError as e:
            logging.critical(str(e))
            sys.exit(EXIT_CODE_CLI_ERROR)
    else:
        logging.info(f"[CACHE] Found {model_name} in cache at {filename}")

    logging.info(f"[VERIFY] Checking SHA256 for {os.path.basename(filename)}...")
    got_sha256 = sha256sum(filename).lower()
    if got_sha256 != expected_sha256.lower():
        logging.critical(f"FATAL: SHA256 HASH MISMATCH for {filename}!\n  Expected: {expected_sha256}\n  Got:      {got_sha256}")
        sys.exit(EXIT_CODE_INTEGRITY_ERROR)
    logging.info(f"[VERIFY] Scanning archive for safety...")
    scan_nemo_archive(filename, max_unzip)
    logging.info("[VERIFY] Verification successful.")
    return filename, got_sha256

# check if a CUDA GPU is found
def check_gpu_availability():
    try:
        import torch as _torch  # local import for safety when used as a lib
    except Exception:
        logging.info("[SETUP] PyTorch not available; skipping GPU check.")
        return
    logging.info("[SETUP] Checking for available NVIDIA GPU...")
    if not _torch.cuda.is_available():
        logging.info("--> No NVIDIA GPU detected.")
    else:
        for i in range(_torch.cuda.device_count()):
            props = _torch.cuda.get_device_properties(i)
            # This is the corrected line. No more duplicates.
            logging.info(f"  - Found Device {i}: {props.name} ({round(props.total_memory / (1024**3), 2)} GB VRAM)")

def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None or shutil.which("ffmpeg.exe") is not None

def preprocess_audio(audio_path, output_dir, require_ffmpeg, workdir, allow_outside):
    if not HAVE_PYDUB:
        if require_ffmpeg:
            logging.critical("FATAL: --require-ffmpeg was set, but `pydub` is not installed.")
            sys.exit(EXIT_CODE_CLI_ERROR)
        logging.warning("`pydub` not installed. Assuming input is already 16kHz mono WAV.")
        if os.path.splitext(audio_path)[1].lower() not in (".wav",):
            logging.warning("Input is not .wav; without ffmpeg/pydub, conversion will be skipped and may fail later.")
        return audio_path
    if not _ffmpeg_available():
        if require_ffmpeg:
            logging.critical("FATAL: --require-ffmpeg was set, but `ffmpeg` was not found in PATH.")
            sys.exit(EXIT_CODE_CLI_ERROR)
        logging.warning("`ffmpeg` not found in PATH. Assuming input is already 16kHz mono WAV.")
        return audio_path
    try:
        logging.info(f"[PREP] Loading audio file: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        if audio.frame_rate == 16000 and audio.channels == 1:
            logging.info("[PREP] Audio is already in the correct 16kHz mono format.")
            return audio_path
        logging.warning(f"[PREP] Audio is not 16kHz mono ({audio.frame_rate} Hz, {audio.channels}ch). Converting...")
        audio = audio.set_frame_rate(16000).set_channels(1)
        converted_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_16khz_mono.wav")
        os.makedirs(output_dir, exist_ok=True)
        if not allow_outside: _must_be_inside_workdir(workdir, converted_path, "converted audio file")
        logging.info(f"[PREP] Exporting converted audio to: {converted_path}")
        audio.export(converted_path, format="wav")
        return converted_path
    except Exception as e:
        logging.error(f"FATAL: Failed to process audio file '{audio_path}': {e}", exc_info=True)
        logging.critical("Please ensure ffmpeg is installed and the audio file is valid.")
        sys.exit(EXIT_CODE_PIPELINE_ERROR)

def create_manifest(audio_filepath, num_speakers, temp_dir, workdir, allow_outside):
    manifest_path = os.path.join(temp_dir, "temp_manifest.json")
    if not allow_outside: _must_be_inside_workdir(workdir, manifest_path, "manifest file")
    meta = {'audio_filepath': os.path.abspath(audio_filepath), 'offset': 0, 'duration': None, 'label': 'infer', 'text': '-', 'rttm_filepath': None, 'uem_filepath': None}
    if num_speakers > 0: meta['num_speakers'] = num_speakers
    with open(manifest_path, 'w', encoding='utf-8') as f: f.write(json.dumps(meta) + '\n')
    logging.info(f"[PREP] Created temporary manifest at: {manifest_path}")
    return manifest_path

def _segments_from_sentences(data, default_dur=3.0, min_span=0.2):
    segs, starts, texts = [], [], []
    lines = [ln for ln in data.get('sentences', '').strip().split('\n') if ln]
    is_debug = logging.getLogger().isEnabledFor(logging.DEBUG)
    for i, ln in enumerate(lines):
        parts = ln.split(' ', 4)
        if len(parts) >= 5:
            try:
                t = parts[0].strip()
                if (t.startswith('[') and t.endswith(']')) or (t.startswith('(') and t.endswith(')')): t = t[1:-1]
                starts.append(float(t))
                texts.append(parts[4].strip())
            except (ValueError, IndexError):
                if is_debug: logging.debug(f"Skipped malformed sentence line in transcript (index {i}): {ln}")
                continue
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else s + default_dur
        if e <= s: e = s + min_span
        segs.append({'start': s, 'end': e, 'text': texts[i]})
    return segs

def format_transcript(transcript_path, output_format):
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f: data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Error reading transcript file: {e}"); return f"Could not generate {output_format} transcript."
    if output_format == 'json': return json.dumps(data, indent=2)
    if output_format in ('srt', 'vtt'):
        segs = data.get('segments')
        if not segs and 'sentences' in data:
            logging.info("No 'segments' found, creating subtitle timings from 'sentences'.")
            segs = _segments_from_sentences(data)
        if not segs:
            logging.warning("No subtitle segments available after fallback; emitting empty file.")
            return "WEBVTT\n\n" if output_format == 'vtt' else ""
        def fmt_time(t_sec, vtt=False):
            total_ms = int(round(t_sec * 1000))
            h, rem = divmod(total_ms, 3600_000)
            m, rem = divmod(rem, 60_000)
            s, ms = divmod(rem, 1000)
            sep = "." if vtt else ","
            return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"
        lines, idx, min_span = [], 1, 0.2
        for segment in (segs or []):
            start, end, text = float(segment['start']), float(segment['end']), (segment.get('text') or '').strip()
            if end - start < 0.001: continue
            if end <= start: end = start + min_span
            line = f"{fmt_time(start, vtt=(output_format=='vtt'))} --> {fmt_time(end, vtt=(output_format=='vtt'))}\n{text}\n"
            if output_format == 'srt': line = f"{idx}\n" + line
            lines.append(line)
            idx += 1
        body = "\n".join(lines)
        return "WEBVTT\n\n" + body if output_format == 'vtt' else body
    lines = []
    if 'sentences' in data:
        for sentence in data['sentences'].strip().split('\n'):
            parts = sentence.split(' ', 4)
            if len(parts) >= 5:
                t = parts[0].strip()
                if (t.startswith('[') and t.endswith(']')) or (t.startswith('(') and t.endswith(')')): t = t[1:-1]
                start_seconds = float(t)
                speaker, text = parts[3].replace(':', ''), parts[4]
                h, m, s = int(start_seconds // 3600), int((start_seconds % 3600) // 60), int(start_seconds % 60)
                time_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
                lines.append(f"=== {time_str} ({speaker}) ===\n{text}\n")
    return "\n".join(lines) if lines else "Transcription complete, but no sentences were found."

def _pick_device(dev_arg_norm: str) -> str:
    if dev_arg_norm == "auto":
        try:
            import torch  # local import so this works when used as a lib
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return dev_arg_norm

def _is_inside(parent, child):
    try:
        return os.path.commonpath([os.path.abspath(parent)]) == os.path.commonpath([os.path.abspath(parent), os.path.abspath(child)])
    except Exception:
        return False

def _is_inside_workdir(parent, child):
    try:
        p = os.path.abspath(parent)
        c = os.path.abspath(child)
        return os.path.commonpath([p]) == os.path.commonpath([p, c])
    except Exception: return False

def _must_be_inside_workdir(base, path, label):
    if not _is_inside_workdir(base, path):
        logging.critical(f"FATAL: Refusing to write {label} outside --workdir: {path}")
        sys.exit(EXIT_CODE_CLI_ERROR)

def _force_no_overlap(cfg):
    # Print to confirm what's actually effective
    try:
        from omegaconf import OmegaConf
        logging.info("[DEBUG] Pre-clamp VAD cfg:\n%s", OmegaConf.to_yaml(cfg.diarizer.vad))
    except Exception:
        pass

    # Write 0.0 to every place NeMo might read from (flat + nested)
    try: cfg.diarizer.vad.overlap = 0.0
    except Exception: pass
    try: cfg.diarizer.vad.parameters.overlap = 0.0
    except Exception: pass

    # Make sure we’re not triggering the overlap path accidentally
    for k, v in {
        "smoothing": "median",
        "window_length_in_sec": 0.20,
        "shift_length_in_sec": 0.02,
        "onset": 0.25,
        "offset": 0.25,
        "min_duration_on": 0.05,
        "min_duration_off": 0.10,
        "pad_onset": 0.05,
        "pad_offset": 0.05,
        "filter_speech_first": True,
        "threshold": 0.25,
    }.items():
        try: setattr(cfg.diarizer.vad, k, v)
        except Exception: pass
        try: setattr(cfg.diarizer.vad.parameters, k, v)
        except Exception: pass

    # Log the final effective VAD section
    try:
        from omegaconf import OmegaConf
        logging.info("[DEBUG] Post-clamp VAD cfg:\n%s", OmegaConf.to_yaml(cfg.diarizer.vad))
    except Exception:
        pass

def main(args):
    if not args.no_asr: _assert_transformers_version()

    # >>> ensure these imports exist even when not run via __main__ <<<
    try:
        import torch
        from omegaconf import OmegaConf as om
    except Exception as e:
        logging.critical(f"FATAL: required libraries missing at runtime: {e}")
        sys.exit(EXIT_CODE_CLI_ERROR)

    # MSDD toggle
    msdd_disabled = str(getattr(args, "msdd_model", "")).strip().lower() in {"", "none", "off", "false"}

    if args.dry_run:
        logging.info("[DRY-RUN] Validating arguments without creating artifacts...")
        if args.output_dir and not args.allow_outside_workdir and not _is_inside_workdir(args.workdir, args.output_dir):
            logging.critical("FATAL: --dry-run: --output-dir is outside --workdir.")
            sys.exit(EXIT_CODE_CLI_ERROR)
        if args.output_filepath and not args.allow_outside_workdir and not _is_inside_workdir(args.workdir, args.output_filepath):
            logging.critical("FATAL: --dry-run: --output-filepath is outside --workdir.")
            sys.exit(EXIT_CODE_CLI_ERROR)
        for k, v in {k:v for k,v in vars(args).items() if k.endswith('_model')}.items():
            if k == "msdd_model" and msdd_disabled:
                continue
            if v and not os.path.isfile(v) and v not in KNOWN_MODELS:
                logging.critical(f"FATAL: --dry-run: Unknown model '{v}' for --{k.replace('_','-')}.")
                sys.exit(EXIT_CODE_CLI_ERROR)
        if args.require_ffmpeg and not _ffmpeg_available():
            logging.critical("FATAL: --dry-run: --require-ffmpeg set, but ffmpeg not found in PATH.")
            sys.exit(EXIT_CODE_CLI_ERROR)
        logging.info("[DRY-RUN] Dry run validation successful.")
        return

    if not os.path.isfile(args.audio_filepath):
        logging.critical(f"FATAL: Audio file not found: {args.audio_filepath}")
        sys.exit(EXIT_CODE_CLI_ERROR)

    dev_norm = args.device.lower()
    if dev_norm == 'cuda' or dev_norm.startswith("cuda:"):
        if not torch.cuda.is_available():
            logging.critical(f"FATAL: Requested device '{args.device}' but CUDA is not available.")
            sys.exit(EXIT_CODE_CLI_ERROR)
        if dev_norm.startswith("cuda:"):
            try:
                idx = int(dev_norm.split(":",1)[1].strip())
                if idx < 0: raise ValueError("CUDA device index must be >= 0")
                if idx >= torch.cuda.device_count():
                    logging.critical(f"FATAL: Requested {args.device}, but only {torch.cuda.device_count()} CUDA device(s) are available.")
                    sys.exit(EXIT_CODE_CLI_ERROR)
            except (ValueError, IndexError):
                logging.critical(f"FATAL: Invalid CUDA device format: '{args.device}'. Use 'cuda' or 'cuda:N'.")
                sys.exit(EXIT_CODE_CLI_ERROR)
    device = _pick_device(dev_norm)
    logging.info(f"[SETUP] Using device: {device}")

    if device.startswith("cuda:"):
        try:
            import torch
            torch.cuda.set_device(int(device.split(":", 1)[1]))
        except Exception:
            pass

    if device == 'cpu':
        logging.warning("--> Running on CPU. Expect slow performance. Consider '--device cuda' with a CUDA-enabled torch.")
    elif device.startswith('mps'):
        logging.critical("FATAL: Apple MPS is not supported for NeMo diarization pipelines. Use CPU or CUDA.")
        sys.exit(EXIT_CODE_CLI_ERROR)
    if args.deterministic:
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if dev_norm.startswith("cuda") and torch.cuda.is_available():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.manual_seed(args.seed)
    try:
        torch.set_num_threads(min(os.cpu_count() or 4, 8))
    except Exception:
        pass

    processed_audio_path = preprocess_audio(args.audio_filepath, args.output_dir, args.require_ffmpeg, args.workdir, args.allow_outside_workdir)

    model_cache_dir = os.path.join(args.cache_root, "nemo_models")
    model_artifacts = {}
    model_args = {k: v for k, v in vars(args).items() if k.endswith('_model')}
    for key, name_or_path in model_args.items():
        if key == "msdd_model" and msdd_disabled:
            continue
        if name_or_path:
            path, sha = fetch_and_verify_model(name_or_path, model_cache_dir, args.allow_online, args.no_progress)
            model_artifacts[key] = {'path': path, 'sha256': sha}
    model_paths = {k: v['path'] for k, v in model_artifacts.items()}

    required_models = ["vad_model", "embedding_model"] + ([] if msdd_disabled else ["msdd_model"])
    if not args.no_asr: required_models.append("asr_model")
    missing = [k for k in required_models if k not in model_paths]
    if missing:
        logging.critical(f"FATAL: Missing required models for the selected mode: {', '.join(missing)}")
        sys.exit(EXIT_CODE_CLI_ERROR)

    # safe to log now
    if msdd_disabled:
        logging.info(
            "[MODELS] Using: VAD=%s | Embeddings=%s%s",
            os.path.basename(model_paths['vad_model']),
            os.path.basename(model_paths['embedding_model']),
            "" if args.no_asr else f" | ASR={os.path.basename(model_paths['asr_model'])}",
        )
    else:
        logging.info(
            "[MODELS] Using: VAD=%s | Embeddings=%s | MSDD=%s%s",
            os.path.basename(model_paths['vad_model']),
            os.path.basename(model_paths['embedding_model']),
            os.path.basename(model_paths['msdd_model']),
            "" if args.no_asr else f" | ASR={os.path.basename(model_paths['asr_model'])}",
        )

    os.environ.setdefault("TORCH_HOME", os.path.join(args.cache_root, "torch"))
    os.environ.setdefault("HF_HOME", os.path.join(args.cache_root, "hf"))
    if not args.allow_online:
        logging.info("[SETUP] Network access disabled; forcing HuggingFace libs to run offline.")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    elif os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        logging.warning("HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1 is set in the environment; this may block model downloads despite --allow-online.")

    logging.info("[PIPELINE] Configuring NeMo pipeline...")

    # Pick the correct diarizer class for the installed NeMo
    def _get_diarizer_class(msdd_enabled: bool):
        if msdd_enabled:
            # MSDD trajectory model diarizer
            try:
                from nemo.collections.asr.models.msdd_models import NeuralDiarizer
                return NeuralDiarizer, "NeuralDiarizer (MSDD)"
            except Exception:
                from nemo.collections.asr.models import NeuralDiarizer  # type: ignore
                return NeuralDiarizer, "NeuralDiarizer (MSDD)"
        else:
            # Clustering-only diarizer
            try:
                from nemo.collections.asr.models import ClusteringDiarizer
                return ClusteringDiarizer, "ClusteringDiarizer"
            except Exception:
                from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer  # type: ignore
                return ClusteringDiarizer, "ClusteringDiarizer"

    # Build config (include keys some NeMo versions try to set, to avoid struct errors)
    manifest_fp = create_manifest(processed_audio_path, args.num_speakers, args.output_dir, args.workdir, args.allow_outside_workdir)

    diarizer_config = {
        'manifest_filepath': manifest_fp,        # also inside 'diarizer'
        'out_dir': args.output_dir,              # also inside 'diarizer'
        'oracle_vad': False,                     # prevent "Key 'oracle_vad' is not in struct"

        # These two are in NeMo’s reference configs and are harmless to set:
        'collar': 0.25,
        'ignore_overlap': True,

        'vad': {'model_path': model_paths['vad_model']}, # for non-strict NeMo variants

        # # 'vad': {'model_path': model_paths['vad_model']}, # for non-strict NeMo variants
        # 'vad': {
        #             'model_path': model_paths['vad_model'],
        #             'parameters': {
        #                 # --- Old parameters that are STILL REQUIRED ---
        #                 'window_length_in_sec': 0.15,
        #                 'shift_length_in_sec': 0.01,
        #                 'threshold': 0.5,

        #                 # --- New parameters for post-processing that are ALSO REQUIRED ---
        #                 'smoothing': 'median',
        #                 'overlap': 0.5,
        #                 'onset': 0.1,
        #                 'offset': 0.1,
        #                 'pad_onset': 0.05,
        #                 'pad_offset': -0.05,
        #                 'min_duration_on': 0.1,
        #                 'min_duration_off': 0.2
        #             }
        #         },

        # 'vad': {
        #     'model_path': model_paths['vad_model'],
        #     'parameters': {
        #         'window_length_in_sec': 0.15,
        #         'shift_length_in_sec': 0.01,
        #         'smoothing': 'median',
        #         'overlap': 0.5,
        #         'onset': 0.1,
        #         'offset': 0.1,
        #         'pad_onset': 0.1,
        #         'pad_offset': 0.0,
        #         'min_duration_on': 0.0,
        #         'min_duration_off': 0.2,
        #         'filter_speech_first': True
        #     }
        # },

        # # 'speaker_embeddings': {'model_path': model_paths['embedding_model']},
        # 'speaker_embeddings': {
        #     'model_path': model_paths['embedding_model'],
        #     'parameters': {
        #         'window_length_in_sec': 1.5,
        #         'shift_length_in_sec': 0.75,
        #         'multiscale_weights': [1, 1, 1],
        #         'save_embeddings': False,
        #         'batch_size': 32
        #     }
        # },        

        # 'speaker_embeddings': {
        #     'model_path': model_paths['embedding_model'],
        #     'parameters': {
        #         'window_length_in_sec': [1.5, 1.25, 1.0, 0.75, 0.5],
        #         'shift_length_in_sec':  [0.75, 0.625, 0.5, 0.375, 0.25],
        #         'multiscale_weights':   [1, 1, 1, 1, 1],
        #         'save_embeddings': False
        #     }
        # },

        'speaker_embeddings': {
            'model_path': model_paths['embedding_model'],
        },

        'clustering': {}
    }
    
    if not msdd_disabled:
        diarizer_config['msdd_model'] = {'model_path': model_paths['msdd_model']}
    else:
        # Expose useful clustering params from NeMo
        diarizer_config['clustering']['parameters'] = {
            'oracle_num_speakers': bool(args.num_speakers > 0),
            'max_num_speakers': args.max_speakers,
            'max_rp_threshold': args.cluster_threshold,
            'maj_vote_spk_count': args.maj_vote_spk_count,
        }
        logging.info("[PIPELINE] MSDD disabled; using clustering-only diarization.")

    # if not args.no_asr and 'asr_model' in model_paths:
    #     diarizer_config['asr'] = {'model_path': model_paths['asr_model']}
    # if args.no_asr:
    #     logging.info("[PIPELINE] ASR disabled (--no-asr). Expecting RTTM output only.")

    if not args.no_asr and 'asr_model' in model_paths:
            if msdd_disabled:
                logging.warning("[PIPELINE] ASR model was provided, but it will be ignored because the clustering-only diarizer (no MSDD) does not support transcription.")
            else:
                # Only add the ASR model if we're using a diarizer that supports it (i.e., MSDD is enabled)
                diarizer_config['asr'] = {'model_path': model_paths['asr_model']}

    # # # // old cfg
    # # cfg = om.create({
    # #     # Some versions read these at top-level:
    # #     'manifest_filepath': manifest_fp,
    # #     'out_dir': args.output_dir,
    # #     'oracle_num_speakers': args.num_speakers if args.num_speakers > 0 else -1,
    # #     # Most versions expect the nested dict:
    # #     'diarizer': diarizer_config
    # # })

    # cfg = om.create({
    #         # Some versions read these at top-level:
    #         'manifest_filepath': manifest_fp,
    #         'out_dir': args.output_dir,
    #         'oracle_num_speakers': args.num_speakers if args.num_speakers > 0 else -1,
    #         'device': device,
    #         'num_workers': 0,
    #         'sample_rate': 16000,
    #         'verbose': False, # Add this line to satisfy the diarizer's internal code
    #         'batch_size': 32,            
    #         # Most versions expect the nested dict:
    #         'diarizer': diarizer_config
    #     })

    # --- START: FIXED CONFIG ---

    # # Re-create manifest path (safe to do again here)
    # manifest_fp = create_manifest(
    #     processed_audio_path, args.num_speakers, args.output_dir, args.workdir, args.allow_outside_workdir
    # )

    # # minimal NeMo cfg dict
    # cfg_dict = {
    #     'manifest_filepath': manifest_fp,
    #     'out_dir': args.output_dir,
    #     'oracle_num_speakers': args.num_speakers if args.num_speakers > 0 else -1,
    #     'sample_rate': 16000,
    #     'num_workers': 0,
    #     'batch_size': 32,
    #     'device': device,
    #     'verbose': False,

    #     'diarizer': {
    #         'manifest_filepath': manifest_fp,
    #         'out_dir': args.output_dir,
    #         'oracle_vad': False,
    #         'collar': 0.25,
    #         'ignore_overlap': True,

    #         # --- START: DEFINITIVE FLAT VAD CONFIG ---
    #         'vad': {
    #             'model_path': model_paths['vad_model'],

    #             # All parameters are now FLATTENED (no nested 'parameters' dict)
    #             # and use robust values from NVIDIA's official recipes.
    #             'window_length_in_sec': 0.15,
    #             'shift_length_in_sec': 0.01,
    #             'smoothing': 'median',
    #             'overlap': 0.875,
    #             'onset': 0.8,           # A higher threshold for speech start can improve accuracy
    #             'offset': 0.5,          # A standard threshold for speech end
    #             'pad_onset': 0.1,
    #             'pad_offset': -0.05,
    #             'min_duration_on': 0.1,
    #             'min_duration_off': 0.2,
    #         },
    #         # --- END: DEFINITIVE FLAT VAD CONFIG ---

    #         'speaker_embeddings': {
    #             'model_path': model_paths['embedding_model'],
    #             'parameters': {
    #                 'window_length_in_sec': [1.5, 1.25, 1.0, 0.75, 0.5],
    #                 'shift_length_in_sec':  [0.75, 0.625, 0.5, 0.375, 0.25],
    #                 'multiscale_weights':   [1, 1, 1, 1, 1],
    #                 'save_embeddings': False,
    #             }
    #         },

    #         'clustering': {
    #             'parameters': {
    #                 'oracle_num_speakers': bool(args.num_speakers > 0),
    #                 'max_num_speakers': args.max_speakers,
    #                 'max_rp_threshold': args.cluster_threshold,
    #                 'maj_vote_spk_count': args.maj_vote_spk_count,
    #                 'enhanced_count_thres': 80,
    #                 'sparse_search_volume': 30,
    #             }
    #         }
    #     }
    # }

    # # # Build a dict that matches NeMo’s expected structure 1:1
    # # cfg_dict = {
    # #     # Some builds read these at top-level too
    # #     'manifest_filepath': manifest_fp,
    # #     'out_dir': args.output_dir,
    # #     'oracle_num_speakers': args.num_speakers if args.num_speakers > 0 else -1,
    # #     'sample_rate': 16000,
    # #     'num_workers': 0,
    # #     'batch_size': 32,

    # #     'device': device,
    # #     'verbose': False,

    # #     'diarizer': {
    # #         'manifest_filepath': manifest_fp,
    # #         'out_dir': args.output_dir,
    # #         'oracle_vad': False,
    # #         'collar': 0.25,
    # #         'ignore_overlap': True,

    # #         # 'vad': {
    # #         #     'model_path': model_paths['vad_model'],
    # #         #     'parameters': {
    # #         #         # REQUIRED frame params (fixes empty TensorList crash)
    # #         #         'window_length_in_sec': 0.15,
    # #         #         'shift_length_in_sec': 0.01,

    # #         #         # Post-processing params that NeMo expects
    # #         #         'smoothing': 'median',
    # #         #         'overlap': 0.875,
    # #         #         'onset': 0.4,
    # #         #         'offset': 0.7,
    # #         #         'pad_onset': 0.05,
    # #         #         'pad_offset': -0.1,
    # #         #         'min_duration_on': 0.2,
    # #         #         'min_duration_off': 0.2,
    # #         #         'filter_speech_first': True,
    # #         #     }
    # #         # },

    # #         # 'vad': {
    # #         # 'model_path': model_paths['vad_model'],
    # #         # 'parameters': {
    # #         #     'window_length_in_sec': 0.15,
    # #         #     'shift_length_in_sec': 0.01,
    # #         #     'smoothing': 'median',
    # #         #     'overlap': 0.875,
    # #         #     'onset': 0.4,
    # #         #     'offset': 0.7,
    # #         #     'pad_onset': 0.05,   # was 0.05 or 0.1 before—ok either way
    # #         #     'pad_offset': 0.0,   # <-- change from -0.1
    # #         #     'min_duration_on': 0.05,  # <-- change from 0.2 / 0.0
    # #         #     'min_duration_off': 0.1,  # <-- change from 0.2
    # #         #     'filter_speech_first': True
    # #         # }
    # #         # },

    # #         # 'vad': {
    # #         #     'model_path': model_paths['vad_model'],
    # #         #     # IMPORTANT: these two belong here, not inside "parameters"
    # #         #     'window_length_in_sec': 0.15,
    # #         #     'shift_length_in_sec': 0.01,

    # #         #     'parameters': {
    # #         #         # keep post-proc thresholds here
    # #         #         'onset': 0.4,
    # #         #         'offset': 0.7,
    # #         #         'pad_onset': 0.05,
    # #         #         'pad_offset': 0.0,
    # #         #         'min_duration_on': 0.05,
    # #         #         'min_duration_off': 0.1,

    # #         #         # optional extras you already had; harmless to keep
    # #         #         'smoothing': 'median',
    # #         #         'overlap': 0.875,
    # #         #         'filter_speech_first': True,
    # #         #     }
    # #         # },

    # #         # # 'vad': {'model_path': model_paths['vad_model']},

    # #         # 'vad': {
    # #         #     'model_path': model_paths['vad_model'],
    # #         #     'parameters': {
    # #         #         'window_length_in_sec': 0.15,
    # #         #         'shift_length_in_sec': 0.01,
    # #         #         'smoothing': 'median',
    # #         #         'overlap': 0.875,
    # #         #         'onset': 0.4,
    # #         #         'offset': 0.7,
    # #         #         'pad_onset': 0.05,
    # #         #         'pad_offset': -0.1,
    # #         #         'min_duration_on': 0.2,
    # #         #         'min_duration_off': 0.2,
    # #         #         'filter_speech_first': True,
    # #         #     },
    # #         # },            

    # #         # Corrected and FLATTENED VAD configuration
    # #         'vad': {
    # #             'model_path': model_paths['vad_model'],

    # #             # NOTE: All parameters are now at the same level as 'model_path'.
    # #             # There is NO nested 'parameters' dictionary.
                
    # #             # Core framing parameters
    # #             'window_length_in_sec': 0.15,
    # #             'shift_length_in_sec': 0.01,

    # #             # Post-processing and thresholding
    # #             'smoothing': 'median',
    # #             'overlap': 0.5,
    # #             'onset': 0.5,          # Threshold for starting speech
    # #             'offset': 0.5,         # Threshold for ending speech
    # #             'pad_onset': 0.1,      # Add 100ms padding to the start of a speech segment
    # #             'pad_offset': 0.0,     # No padding at the end
    # #             'min_duration_on': 0.1,
    # #             'min_duration_off': 0.2,
    # #         },

    # #         'speaker_embeddings': {
    # #             'model_path': model_paths['embedding_model'],
    # #             'parameters': {
    # #                 # Multiscale settings used by NeMo diarization recipes
    # #                 'window_length_in_sec': [1.5, 1.25, 1.0, 0.75, 0.5],
    # #                 'shift_length_in_sec':  [0.75, 0.625, 0.5, 0.375, 0.25],
    # #                 'multiscale_weights':   [1, 1, 1, 1, 1],
    # #                 'save_embeddings': False,
    # #             }
    # #         },

    # #         'clustering': {
    # #             'parameters': {
    # #                 'oracle_num_speakers': bool(args.num_speakers > 0),
    # #                 'max_num_speakers': args.max_speakers,
    # #                 'max_rp_threshold': args.cluster_threshold,
    # #                 'maj_vote_spk_count': args.maj_vote_spk_count,
    # #                 # Helpful, sane defaults
    # #                 'enhanced_count_thres': 80,
    # #                 'sparse_search_volume': 30,
    # #             }
    # #         }
    # #     }
    # # }

    # # # Add MSDD/ASR only when applicable
    # # # if not msdd_disabled:
    # # #     cfg_dict['diarizer']['msdd_model'] = {'model_path': model_paths['msdd_model']}
    # # #     if not args.no_asr and 'asr_model' in model_paths:
    # # #         cfg_dict['diarizer']['asr'] = {'model_path': model_paths['asr_model']}
    # # # elif not args.no_asr:
    # # #     logging.warning("[PIPELINE] ASR model was provided, but it will be ignored because the clustering-only diarizer (no MSDD) does not support transcription.")

    # # if not msdd_disabled:
    # #     cfg_dict['diarizer']['msdd_model'] = {'model_path': model_paths['msdd_model']}

    # # if not args.no_asr and 'asr_model' in model_paths:
    # #     cfg_dict['diarizer']['asr'] = {'model_path': model_paths['asr_model']}

    # # Add MSDD/ASR only when applicable
    # if not msdd_disabled:
    #     cfg_dict['diarizer']['msdd_model'] = {'model_path': model_paths['msdd_model']}
    #     if not args.no_asr and 'asr_model' in model_paths:
    #         cfg_dict['diarizer']['asr'] = {'model_path': model_paths['asr_model']}
    # elif not args.no_asr:
    #     logging.warning("[PIPELINE] ASR model was provided, but it will be ignored because the clustering-only diarizer (no MSDD) does not support transcription.")

    # Re-create manifest path (safe to do again here)
    manifest_fp = create_manifest(
        processed_audio_path, args.num_speakers, args.output_dir, args.workdir, args.allow_outside_workdir
    )

    # Build a dict that matches NeMo’s expected structure 1:1
    cfg_dict = {
        'manifest_filepath': manifest_fp,
        'out_dir': args.output_dir,
        'oracle_num_speakers': args.num_speakers if args.num_speakers > 0 else -1,
        'sample_rate': 16000,
        'num_workers': 0,
        'batch_size': 32,
        'device': device,
        'verbose': False,

        'diarizer': {
            'manifest_filepath': manifest_fp,
            'out_dir': args.output_dir,
            'oracle_vad': False,
            'collar': 0.25,
            'ignore_overlap': True,

            # --- START: CORRECT AND FINAL VAD CONFIG ---
            # --- VAD config for NeMo 2.4.x ClusteringDiarizer ---
            'vad': {
                'model_path': model_paths['vad_model'],
                'parameters': {
                    # Frame settings
                    'window_length_in_sec': 0.15,
                    'shift_length_in_sec' : 0.01,

                    # Post-processing
                    'smoothing'          : 'median',
                    'overlap'            : 0.5,
                    'onset'              : 0.1,
                    'offset'             : 0.1,
                    'pad_onset'          : 0.1,
                    'pad_offset'         : 0.0,
                    'min_duration_on'    : 0.0,
                    'min_duration_off'   : 0.2,
                    'filter_speech_first': True,
                },
            },
            # --- END: VAD CONFIG ---

            'speaker_embeddings': {
                'model_path': model_paths['embedding_model'],
                'parameters': {
                    'window_length_in_sec': [1.5, 1.25, 1.0, 0.75, 0.5],
                    'shift_length_in_sec':  [0.75, 0.625, 0.5, 0.375, 0.25],
                    'multiscale_weights':   [1, 1, 1, 1, 1],
                    'save_embeddings': False,
                }
            },

            'clustering': {
                'parameters': {
                    'oracle_num_speakers': bool(args.num_speakers > 0),
                    'max_num_speakers': args.max_speakers,
                    'max_rp_threshold': args.cluster_threshold,
                    'maj_vote_spk_count': args.maj_vote_spk_count,
                    'enhanced_count_thres': 80,
                    'sparse_search_volume': 30,
                }
            }
        }
    }

    # Add MSDD/ASR only when applicable
    if not msdd_disabled:
        cfg_dict['diarizer']['msdd_model'] = {'model_path': model_paths['msdd_model']}
        if not args.no_asr and 'asr_model' in model_paths:
            cfg_dict['diarizer']['asr'] = {'model_path': model_paths['asr_model']}
    elif not args.no_asr:
        logging.warning("[PIPELINE] ASR model was provided, but it will be ignored because the clustering-only diarizer (no MSDD) does not support transcription.")

    # Convert to OmegaConf
    cfg = om.create(cfg_dict)

    # Make the cfg permissive so NeMo’s version-specific schema can merge cleanly
    from omegaconf import OmegaConf as _om

    # Allow NeMo to merge version-specific keys w/o struct errors
    try:
        om.set_struct(cfg, False)
        if 'diarizer' in cfg:
            om.set_struct(cfg.diarizer, False)
            if 'vad' in cfg.diarizer:
                om.set_struct(cfg.diarizer.vad, False)
                if 'parameters' in cfg.diarizer.vad:
                    om.set_struct(cfg.diarizer.vad.parameters, False)
    except Exception:
        pass

    # try:
    #     _om.set_struct(cfg, False)
    #     if 'diarizer' in cfg:
    #         _om.set_struct(cfg.diarizer, False)
    #         if 'vad' in cfg.diarizer:
    #             _om.set_struct(cfg.diarizer.vad, False)
    #             if 'parameters' in cfg.diarizer.vad:
    #                 _om.set_struct(cfg.diarizer.vad.parameters, False)
    # except Exception:
    #     pass

    if args.diarizer_config:
        try:
            override_cfg = om.load(args.diarizer_config)
            cfg = om.merge(cfg, override_cfg)
            logging.info(f"[PIPELINE] Merged diarizer config from {args.diarizer_config}")
        except Exception as e:
            logging.critical(f"FATAL: Could not load or merge diarizer config: {e}")
            sys.exit(EXIT_CODE_CLI_ERROR)

    output_str, success = "", False
    try:
        logging.info("[PIPELINE] Starting NeMo diarization/transcription job...")
        DiarizerClass, diarizer_label = _get_diarizer_class(msdd_enabled=not msdd_disabled)
        logging.info(f"[PIPELINE] Using diarizer class: {diarizer_label}")

        if args.diarizer_config:
            override_cfg = om.load(args.diarizer_config)
            cfg = om.merge(cfg, override_cfg)
            logging.info(f"[PIPELINE] Merged diarizer config from {args.diarizer_config}")

        _force_no_overlap(cfg)
        DiarizerClass, diarizer_label = _get_diarizer_class(msdd_enabled=not msdd_disabled)        
        
        dm = DiarizerClass(cfg=cfg)

        # Some NeMo builds are Lightning modules; .to() may or may not exist—be liberal.
        try:
            dm = dm.to(device)  # type: ignore[attr-defined]
        except Exception:
            pass

        # ClusteringDiarizer typically exposes .diarize(); some MSDD variants add .transcribe()
        if hasattr(dm, "diarize"):
            dm.diarize()  # type: ignore[attr-defined]
        elif hasattr(dm, "transcribe"):
            dm.transcribe()  # type: ignore[attr-defined]
        else:
            dm()  # type: ignore[operator]

        # def _run_once(model):
        #     if hasattr(model, "diarize"): model.diarize()
        #     elif hasattr(model, "transcribe"): model.transcribe()
        #     else: model()

        # try:
        #     _run_once(dm)
        # except Exception as e:
        #     msg = str(e)
        #     # Retry path 1: the overlap bug (we already handled before, but keep it)
        #     if "TensorList" in msg or "stack expects a non-empty TensorList" in msg:
        #         logging.warning("[PIPELINE] Overlap smoother crashed; retrying with overlap=0.0 and median smoothing.")
        #         p = cfg.diarizer.vad.parameters
        #         p.overlap = 0.0
        #         p.smoothing = "median"
        #         p.filter_speech_first = True
        #         dm = DiarizerClass(cfg=cfg)
        #         try: dm = dm.to(device)
        #         except Exception: pass
        #         _run_once(dm)
        #     # Retry path 2: VAD returned zero segments → loosen thresholds
        #     elif "contains silence" in msg.lower():
        #         logging.warning("[PIPELINE] VAD found zero speech; retrying with looser thresholds.")
        #         p = cfg.diarizer.vad.parameters
        #         p.overlap = 0.0
        #         p.smoothing = "median"
        #         p.filter_speech_first = True
        #         p.window_length_in_sec = 0.20
        #         p.shift_length_in_sec  = 0.02
        #         p.onset  = 0.25
        #         p.offset = 0.25
        #         try: p.threshold = 0.25
        #         except Exception: pass
        #         p.min_duration_on  = max(0.05, float(getattr(p, "min_duration_on", 0.0)))
        #         p.min_duration_off = max(0.10, float(getattr(p, "min_duration_off", 0.1)))
        #         p.pad_onset  = 0.05
        #         p.pad_offset = 0.05
        #         dm = DiarizerClass(cfg=cfg)
        #         try: dm = dm.to(device)
        #         except Exception: pass
        #         _run_once(dm)
        #     else:
        #         raise

        logging.info("[PIPELINE] NeMo job finished.")

        rttm_dir = os.path.join(args.output_dir, "pred_rttms")
        rttm_path = os.path.join(rttm_dir, os.path.splitext(os.path.basename(processed_audio_path))[0] + ".rttm")
        if not os.path.exists(rttm_path) and os.path.isdir(rttm_dir):
            try:
                candidates = sorted(glob.glob(os.path.join(rttm_dir, "*.rttm")))
                if len(candidates) == 1: rttm_path = candidates[0]
            except Exception:
                pass
        transcript_path = os.path.join(args.output_dir, "diar_msdd_asr_transcript.json")

        # broadened fallback
        if not os.path.exists(transcript_path):
            try:
                cands = sorted(glob.glob(os.path.join(args.output_dir, "*.json")))
                hints = ("transcript", "asr", "diar", "result")
                ranked = sorted(cands, key=lambda p: (not any(h in os.path.basename(p).lower() for h in hints), p))
                if ranked:
                    transcript_path = ranked[0]
            except Exception:
                pass

        if not args.no_asr and os.path.exists(transcript_path):
            output_str = format_transcript(transcript_path, args.output_format)
        elif os.path.exists(rttm_path):
            if not args.no_asr:
                logging.warning("ASR was requested but no transcript JSON was found; emitting diarization-only RTTM.")
            output_str = f"Diarization-only complete. RTTM file saved to: {rttm_path}"
            if args.output_filepath and args.output_filepath.lower().endswith(".rttm"):
                if not args.allow_outside_workdir: _must_be_inside_workdir(args.workdir, args.output_filepath, "RTTM output file")
                os.makedirs(os.path.dirname(os.path.abspath(args.output_filepath)), exist_ok=True)
                shutil.copyfile(rttm_path, args.output_filepath)
                logging.info(f"RTTM file copied to specified output path: {args.output_filepath}")
            else:
                logging.info(f"RTTM available at default path: {rttm_path}")
        else:
            output_str = "Pipeline finished, but no transcript or RTTM file was found."
        success = True
    except Exception as e:
        logging.critical(f"NeMo pipeline failed: {e}", exc_info=(args.log_level=="DEBUG"))
        sys.exit(EXIT_CODE_PIPELINE_ERROR)
    finally:
        try:
            if args.cleanup_run and _is_inside_workdir(args.workdir, args.output_dir):
                preserve = set()
                if args.output_filepath:
                    preserve.add(os.path.abspath(args.output_filepath))
                try:
                    if os.path.exists(rttm_path):
                        preserve.add(os.path.abspath(rttm_path))
                except NameError:
                    pass

                if any(_is_inside(args.output_dir, p) for p in preserve):
                    for root, dirs, files in os.walk(args.output_dir, topdown=False):
                        for name in files:
                            fp = os.path.join(root, name)
                            if os.path.abspath(fp) not in preserve:
                                try: os.remove(fp)
                                except Exception: pass
                        for name in dirs:
                            dp = os.path.join(root, name)
                            try: os.rmdir(dp)
                            except Exception: pass
                    logging.info(f"Cleanup requested: removed intermediates in {args.output_dir} (preserved final outputs).")
                else:
                    logging.info(f"Cleanup requested: removing run directory {args.output_dir}")
                    shutil.rmtree(args.output_dir)
        except Exception as e:
            logging.warning(f"Automatic cleanup skipped: {e}")

    if args.output_format == 'text' and not args.no_asr: logging.info("=== Diarization and Transcription Output ===")
    print(output_str)
    if args.output_filepath and not args.no_asr:
        if not args.allow_outside_workdir: _must_be_inside_workdir(args.workdir, args.output_filepath, "final transcript")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_filepath)), exist_ok=True)
        with open(args.output_filepath, 'w', encoding='utf-8') as f: f.write(output_str)
        logging.info(f"\nFormatted output has been saved to {args.output_filepath}")

def _ver_tuple_relaxed(s):
    nums = [int(x) for x in re.findall(r'\d+', s)]
    return tuple(nums + [0]*(4-len(nums)))[:4]

def _assert_transformers_version():
    try: v = _im.version('transformers')
    except _im.PackageNotFoundError:
        logging.critical("FATAL: transformers not installed but is required for ASR. Use --no-asr to run diarization only, or install it.")
        sys.exit(EXIT_CODE_CLI_ERROR)
    if _ver_tuple_relaxed(v) < _ver_tuple_relaxed("4.53.0"):
        logging.critical(f"FATAL: transformers=={v} is below the safe floor 4.53.0 (older stacks include security risks).\n"
                         f"Fix: pip install --upgrade \"transformers>=4.53.0\"")
        sys.exit(EXIT_CODE_CLI_ERROR)

class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter): pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=SCRIPT_BANNER, formatter_class=CustomHelpFormatter)
    parser.add_argument("audio_filepath", nargs='?', default=None, help="Path to the audio file to process.")
    parser.add_argument("--workdir", default="./data", help="Root working directory for all runs and artifacts.")
    parser.add_argument("--output-dir", help="Directory for run-specific files. Defaults to a new timestamped folder inside workdir.")
    parser.add_argument("-o", "--output-filepath", help="Optional. Path to save the final transcript or RTTM.")
    parser.add_argument("--num-speakers", type=int, default=0, help="Number of speakers. If 0, model estimates.")
    parser.add_argument("--no-asr", action="store_true", help="Run diarization only (produces RTTM file, no transcript).")
    parser.add_argument("--output-format", default="text", choices=["text","json","srt","vtt"], help="Format for the final transcript.")
    parser.add_argument("--version", action="store_true", help="Print version information and exit.")
    parser.add_argument("--asr-model", default="parakeet-rnnt-1.1b", help="ASR model name or local .nemo path.")
    parser.add_argument("--embedding-model", default="titanet_large", help="Speaker embedding model name or local .nemo path.")
    parser.add_argument("--vad-model", default="vad_multilingual_marblenet", help="VAD model name or local .nemo path.")
    parser.add_argument("--msdd-model", default="none", help="MSDD .nemo path or known id. Use 'none' for clustering-only (default).")
    parser.add_argument("--max-speakers", type=int, default=20, help="Clustering-only: maximum speakers to consider (ignored if --num-speakers>0).")
    parser.add_argument("--cluster-threshold", type=float, default=0.25, help="Clustering-only: upper bound of RP threshold search.")
    parser.add_argument("--maj-vote-spk-count", action="store_true", help="Clustering-only: enable majority-vote speaker counting heuristic.")
    parser.add_argument("--allow-online", action="store_true", help="Allow downloading of known models. Default is offline.")
    parser.add_argument("--allow-outside-workdir", action="store_true", help="Permit writing outside --workdir (unsafe).")
    parser.add_argument("--cache-root", default=os.path.join(os.path.expanduser("~"), ".cache"), help="Root directory for all model and library caches.")
    parser.add_argument("--models-manifest", help="Optional JSON file to update the known models manifest.")
    parser.add_argument("--proxy", help="HTTP(S) proxy for downloads, e.g. http://user:pass@host:port")
    parser.add_argument("--device", default="auto", help="Device to run on (e.g., 'cuda', 'cuda:0', 'cpu', 'auto').")
    parser.add_argument("--seed", type=int, default=1337, help="Seed for reproducibility.")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic ops (slower).")
    parser.add_argument("--diarizer-config", help="Path to a YAML/JSON file to override NeMo's internal diarizer config.")
    parser.add_argument("--cleanup-run", action="store_true", help="Delete the run's output directory on successful completion.")
    parser.add_argument("--dry-run", action="store_true", help="Validate environment and arguments, then exit.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    parser.add_argument("--require-ffmpeg", action="store_true", help="Exit if ffmpeg is not found for audio conversion.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()
    user_set_output_format = any(a == "--output-format" or a.startswith("--output-format=") for a in sys.argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.version:
        print(SCRIPT_BANNER)
        print("\n--- Environment ---")
        py_ver, torch_ver, nemo_ver = f"Python {sys.version.split()[0]}", "torch not installed", "nemo_toolkit not installed"
        try: torch_ver = f"torch {_im.version('torch')}"
        except _im.PackageNotFoundError: pass
        try: nemo_ver = f"nemo_toolkit {_im.version('nemo_toolkit')}"
        except _im.PackageNotFoundError: pass
        print(f"{py_ver} | {torch_ver} | {nemo_ver}")
        sys.exit(EXIT_CODE_SUCCESS)

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if args.no_progress:
        os.environ.setdefault("TQDM_DISABLE", "1")
        os.environ.setdefault("DISABLE_TQDM", "1")
    if args.proxy:
        os.environ["HTTP_PROXY"] = os.environ["http_proxy"] = args.proxy
        os.environ["HTTPS_PROXY"] = os.environ["https_proxy"] = args.proxy
        if no_proxy := os.environ.get("NO_PROXY"): os.environ["no_proxy"] = no_proxy

    # fail fast if `--asr-model` is supplied with `--no-asr`
    if args.no_asr and any(s.startswith("--asr-model") for s in sys.argv):
        logging.critical("FATAL: --asr-model cannot be used with --no-asr.")
        sys.exit(EXIT_CODE_CLI_ERROR)

    if args.no_progress:
        logging.info("Progress bars disabled.")

    _install_signal_handlers()

    try:
        import torch
        from omegaconf import OmegaConf as om
        torch_ver = _im.version('torch')
        def _tup(s):
            nums = [int(x) for x in re.findall(r'\d+', s)]
            return tuple(nums + [0]*(3-len(nums)))[:3]
        if _tup(torch_ver) < _tup("2.0.0"):
            print(f"FATAL: torch=={torch_ver} is below the required 2.0.0.", file=sys.stderr)
            sys.exit(EXIT_CODE_CLI_ERROR)
    except _im.PackageNotFoundError:
        print("FATAL: PyTorch or OmegaConf not installed. See header for install command.", file=sys.stderr)
        sys.exit(EXIT_CODE_CLI_ERROR)
    try:
        nemo_ver = _im.version('nemo_toolkit')
        ok = True
        if _V: ok = _V(nemo_ver) >= _V("2.4.0")
        else:
            def _tup(s):
                nums = [int(x) for x in re.findall(r'\d+', s)]
                return tuple(nums + [0]*(3-len(nums)))[:3]
            ok = _tup(nemo_ver) >= _tup("2.4.0")
        if not ok:
            print(f"FATAL: nemo_toolkit=={nemo_ver} is below the required 2.4.0.", file=sys.stderr)
            sys.exit(EXIT_CODE_CLI_ERROR)
    except _im.PackageNotFoundError:
        print("FATAL: NVIDIA NeMo Toolkit not found. See header for install command.", file=sys.stderr)
        sys.exit(EXIT_CODE_CLI_ERROR)
    except Exception:
        pass

    # if not args.audio_filepath and not args.dry_run: parser.error("the following arguments are required: audio_filepath")

    if args.audio_filepath is None and not args.dry_run:
        print(SCRIPT_BANNER, file=sys.stderr)
        parser.print_usage(sys.stderr)
        sys.exit(EXIT_CODE_CLI_ERROR)

    args.device = (args.device or "").strip()

    if not args.output_dir or args.output_dir == parser.get_default("output_dir"):
        ts, rand = time.strftime("%Y%m%d-%H%M%S"), secrets.token_hex(2)
        args.output_dir = os.path.join(args.workdir, "runs", f"{ts}-{rand}")
    if not args.dry_run:
        if not args.allow_outside_workdir and not _is_inside_workdir(args.workdir, args.output_dir):
            logging.critical(f"FATAL: --output-dir must be inside --workdir. Use --allow-outside-workdir to override.")
            sys.exit(EXIT_CODE_CLI_ERROR)
        os.makedirs(args.workdir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Workdir: {os.path.abspath(args.workdir)}")
        logging.info(f"Run output will be in: {os.path.abspath(args.output_dir)}")
    if not args.dry_run and not args.output_filepath and not args.no_asr:
        base = os.path.splitext(os.path.basename(args.audio_filepath))[0]
        ext = {"text": "txt", "json": "json", "srt": "srt", "vtt": "vtt"}[args.output_format]
        args.output_filepath = os.path.join(args.output_dir, f"{base}.{ext}")
        logging.info(f"No --output-filepath provided; transcript will be saved to {args.output_filepath}")
    elif not args.dry_run and args.no_asr and not args.output_filepath:
        base = os.path.splitext(os.path.basename(args.audio_filepath or "audio"))[0]
        rttm_default = os.path.join(args.output_dir, "pred_rttms", f"{base}.rttm")
        args.output_filepath = rttm_default
        logging.info(f"No --output-filepath provided; diarization-only RTTM will be copied to {args.output_filepath}")
    if args.output_filepath and not args.allow_outside_workdir:
        if not _is_inside_workdir(args.workdir, args.output_filepath):
            logging.critical("FATAL: --output-filepath is outside --workdir. Use --allow-outside-workdir to override.")
            sys.exit(EXIT_CODE_CLI_ERROR)
    if args.no_asr and args.output_filepath and not args.output_filepath.lower().endswith(".rttm"):
        logging.critical("FATAL: --no-asr requires an RTTM output. Please use -o <path>.rttm")
        sys.exit(EXIT_CODE_CLI_ERROR)
    if args.no_asr and user_set_output_format:
        logging.critical("FATAL: --output-format is not applicable with --no-asr (diarization-only). Remove it.")
        sys.exit(EXIT_CODE_CLI_ERROR)
    if not args.dry_run:
        with open(os.path.join(args.output_dir, "RUN.md"), "w", encoding="utf-8") as f:
            f.write(f"# diarize_nemo.py Run Log\n\n- **Date**: {datetime.datetime.now().isoformat()}\n- **Host**: {platform.node()}\n- **Python**: {platform.python_version()}\n")
            try: f.write(f"- **torch**: {_im.version('torch')}\n")
            except _im.PackageNotFoundError: pass
            try: f.write(f"- **nemo_toolkit**: {_im.version('nemo_toolkit')}\n\n")
            except _im.PackageNotFoundError: pass
            f.write("### Command\n```bash\n" + " ".join(map(shlex.quote, sys.argv)) + "\n```\n")
    if args.models_manifest:
        try:
            with open(args.models_manifest, "r", encoding="utf-8") as f:
                new_models = json.load(f)
                _validate_models_manifest(new_models)
                KNOWN_MODELS.update(new_models)
            logging.info(f"Updated known models from {args.models_manifest}")
        except (IOError, json.JSONDecodeError, ValueError) as e:
            logging.critical(f"FATAL: Could not read or parse models manifest file '{args.models_manifest}': {e}")
            sys.exit(EXIT_CODE_INTEGRITY_ERROR)
    if args.no_asr: args.asr_model = None
    elif not args.asr_model: parser.error("--asr-model is required unless --no-asr is specified.")
    try:
        check_gpu_availability()
        main(args)
        sys.exit(EXIT_CODE_SUCCESS)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(EXIT_CODE_CTRL_C)
    except SystemExit as e:
        sys.exit(e.code if e.code is not None and isinstance(e.code, int) else EXIT_CODE_GENERIC_ERROR)
    except Exception as e:
        logging.critical("An unexpected and unhandled error occurred:", exc_info=True)
        sys.exit(EXIT_CODE_GENERIC_ERROR)
