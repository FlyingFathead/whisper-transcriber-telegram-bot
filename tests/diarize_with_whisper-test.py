# diarize_with_whisper-test.py
# requires pyAudioAnalysis and hmmlearn, install with:
# $ pip install pyAudioAnalysis hmmlearn eyed3 imbalanced-learn
#
# given that pyAudioAnalysis runs pickled stuff, you might want to run it as:
# firejail --private python3 diarize_with_whisper-test.py <URL>

import sys
import os
import subprocess
import whisper
from pyAudioAnalysis import audioSegmentation as aS
from datetime import timedelta
import logging
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the model at the start
whisper_model_name = "medium.en"

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Load Whisper model
logging.info("Loading Whisper model...")
whisper_model = whisper.load_model(whisper_model_name).to(device)
logging.info("Whisper model loaded.")

# Transcribe the audio file
def transcribe_audio(file_path):
    logging.info(f"Transcribing audio file: {file_path}")
    result = whisper_model.transcribe(file_path)
    logging.info("Transcription completed.")
    return result['segments']

# Convert Whisper transcription to SRT format
def to_srt(transcription_segments):
    subtitles = []
    for i, segment in enumerate(transcription_segments):
        start = segment['start']
        end = segment['end']
        text = segment['text']
        subtitle = f"{i + 1}\n{format_time(start)} --> {format_time(end)}\n{text}\n"
        subtitles.append(subtitle)
    return "\n".join(subtitles)

def format_time(seconds):
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    milliseconds = seconds % 1
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds * 1000):03}"

# Perform diarization on the audio file using pyAudioAnalysis
def diarize_audio(file_path):
    logging.info(f"Performing diarization on audio file: {file_path}")
    num_speakers = 2  # Assuming 2 speakers, adjust as necessary
    diarization_results = aS.speaker_diarization(file_path, num_speakers)
    diarization_segments = []
    current_speaker = diarization_results[0]
    segment_start = 0.0

    for i, speaker in enumerate(diarization_results):
        if speaker != current_speaker:
            segment_end = i / 100.0  # Assuming a segment resolution of 0.01 seconds
            diarization_segments.append((segment_start, segment_end, current_speaker))
            current_speaker = speaker
            segment_start = segment_end

    segment_end = len(diarization_results) / 100.0
    diarization_segments.append((segment_start, segment_end, current_speaker))

    logging.info("Diarization completed.")
    return diarization_segments

# Parse SRT file to extract transcription segments
def parse_srt(srt_file):
    with open(srt_file, "r") as f:
        lines = f.readlines()
    
    segments = []
    for i in range(0, len(lines), 4):
        start_end = lines[i+1].strip().split(" --> ")
        start_time = parse_time(start_end[0])
        end_time = parse_time(start_end[1])
        text = lines[i+2].strip()
        segments.append((start_time, end_time, text))
    return segments

def parse_time(time_str):
    hours, minutes, seconds = time_str.split(":")
    seconds, milliseconds = seconds.split(",")
    return timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds), milliseconds=int(milliseconds))

# Merge transcription and diarization results
def merge_results(transcription_segments, diarization_segments):
    merged_results = []
    for start_time, end_time, text in transcription_segments:
        speaker_label = "Unknown"
        for diarization_start, diarization_end, speaker in diarization_segments:
            if diarization_start <= start_time.total_seconds() <= diarization_end:
                speaker_label = speaker
                break
        merged_results.append((start_time, end_time, text, speaker_label))
    return merged_results

# Format merged results into the specified plaintext format
def format_merged_results_plaintext(merged_results):
    lines = []
    for start, end, text, speaker in merged_results:
        start_str = format_time(start.total_seconds())
        end_str = format_time(end.total_seconds())
        lines.append(f"=== [ {speaker} | {start_str} - {end_str} ] ===")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)

# Download audio using yt-dlp
def download_audio(url):
    output_file = "downloaded_audio.wav"
    if os.path.exists(output_file):
        overwrite = input(f"{output_file} already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            logging.info("Download aborted by user.")
            sys.exit(1)
    command = ["yt-dlp", "-x", "--audio-format", "wav", "-o", output_file, url]
    logging.info(f"Downloading audio from URL: {url}")
    subprocess.run(command, check=True)
    logging.info("Download completed.")
    return output_file

# Main function to run the whole pipeline
def main(url):
    # Download the audio
    audio_file = download_audio(url)
    
    # Transcribe audio
    transcription_segments = transcribe_audio(audio_file)
    
    # Save initial transcription to SRT (optional)
    srt_output = to_srt(transcription_segments)
    with open("transcription.srt", "w") as f:
        f.write(srt_output)
    
    # Perform diarization
    diarization_segments = diarize_audio(audio_file)
    
    # Parse transcription segments from SRT (if needed)
    transcription_segments = parse_srt("transcription.srt")
    
    # Merge results
    merged_results = merge_results(transcription_segments, diarization_segments)
    
    # Format merged results into plaintext
    merged_plaintext_output = format_merged_results_plaintext(merged_results)
    with open("merged_transcription.txt", "w") as f:
        f.write(merged_plaintext_output)

# Run the main function with the URL provided via the command line
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 diarize_with_whisper-test.py <URL>")
        sys.exit(1)
    url = sys.argv[1]
    main(url)


# === (pyannote.audio version -- requires huggingface API key) ===
# 
# import sys
# import subprocess
# import whisper
# from pyannote.audio import Pipeline
# from datetime import timedelta
# import logging

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Define the model at the start
# whisper_model_name = "medium.en"

# # Load Whisper model
# logging.info("Loading Whisper model...")
# whisper_model = whisper.load_model(whisper_model_name)
# logging.info("Whisper model loaded.")

# # Transcribe the audio file
# def transcribe_audio(file_path):
#     logging.info(f"Transcribing audio file: {file_path}")
#     result = whisper_model.transcribe(file_path)
#     logging.info("Transcription completed.")
#     return result['segments']

# # Convert Whisper transcription to SRT format
# def to_srt(transcription_segments):
#     subtitles = []
#     for i, segment in enumerate(transcription_segments):
#         start = segment['start']
#         end = segment['end']
#         text = segment['text']
#         subtitle = f"{i + 1}\n{format_time(start)} --> {format_time(end)}\n{text}\n"
#         subtitles.append(subtitle)
#     return "\n".join(subtitles)

# def format_time(seconds):
#     hours, seconds = divmod(seconds, 3600)
#     minutes, seconds = divmod(seconds, 60)
#     milliseconds = seconds % 1
#     return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds * 1000):03}"

# # Load the pre-trained diarization pipeline
# logging.info("Loading pyannote diarization pipeline...")
# diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
# logging.info("Diarization pipeline loaded.")

# # Perform diarization on the audio file
# def diarize_audio(file_path):
#     logging.info(f"Performing diarization on audio file: {file_path}")
#     diarization = diarization_pipeline(file_path)
#     diarization_segments = [(segment.start, segment.end, speaker) for segment, _, speaker in diarization.itertracks(yield_label=True)]
#     logging.info("Diarization completed.")
#     return diarization_segments

# # Parse SRT file to extract transcription segments
# def parse_srt(srt_file):
#     with open(srt_file, "r") as f:
#         lines = f.readlines()
    
#     segments = []
#     for i in range(0, len(lines), 4):
#         start_end = lines[i+1].strip().split(" --> ")
#         start_time = parse_time(start_end[0])
#         end_time = parse_time(start_end[1])
#         text = lines[i+2].strip()
#         segments.append((start_time, end_time, text))
#     return segments

# def parse_time(time_str):
#     hours, minutes, seconds = time_str.split(":")
#     seconds, milliseconds = seconds.split(",")
#     return timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds), milliseconds=int(milliseconds))

# # Merge transcription and diarization results
# def merge_results(transcription_segments, diarization_segments):
#     merged_results = []
#     for start_time, end_time, text in transcription_segments:
#         speaker_label = "Unknown"
#         for diarization_start, diarization_end, speaker in diarization_segments:
#             if diarization_start <= start_time.total_seconds() <= diarization_end:
#                 speaker_label = speaker
#                 break
#         merged_results.append((start_time, end_time, text, speaker_label))
#     return merged_results

# # Format merged results into the specified plaintext format
# def format_merged_results_plaintext(merged_results):
#     lines = []
#     for start, end, text, speaker in merged_results:
#         start_str = format_time(start.total_seconds())
#         end_str = format_time(end.total_seconds())
#         lines.append(f"=== [ {speaker} | {start_str} - {end_str} ] ===")
#         lines.append(text)
#         lines.append("")
#     return "\n".join(lines)

# # Download audio using yt-dlp
# def download_audio(url):
#     output_file = "downloaded_audio.wav"
#     command = ["yt-dlp", "-x", "--audio-format", "wav", "-o", output_file, url]
#     logging.info(f"Downloading audio from URL: {url}")
#     subprocess.run(command, check=True)
#     logging.info("Download completed.")
#     return output_file

# # Main function to run the whole pipeline
# def main(url):
#     # Download the audio
#     audio_file = download_audio(url)
    
#     # Transcribe audio
#     transcription_segments = transcribe_audio(audio_file)
    
#     # Save initial transcription to SRT (optional)
#     srt_output = to_srt(transcription_segments)
#     with open("transcription.srt", "w") as f:
#         f.write(srt_output)
    
#     # Perform diarization
#     diarization_segments = diarize_audio(audio_file)
    
#     # Parse transcription segments from SRT (if needed)
#     transcription_segments = parse_srt("transcription.srt")
    
#     # Merge results
#     merged_results = merge_results(transcription_segments, diarization_segments)
    
#     # Format merged results into plaintext
#     merged_plaintext_output = format_merged_results_plaintext(merged_results)
#     with open("merged_transcription.txt", "w") as f:
#         f.write(merged_plaintext_output)

# # Run the main function with the URL provided via the command line
# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python3 diarize_with_whisper-test.py <URL>")
#         sys.exit(1)
#     url = sys.argv[1]
#     main(url)
