# diarization.py
# (From: https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/)

import os
import numpy as np
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from spectralcluster import SpectralClusterer, RefinementOptions
import librosa
import whisper
from pydub import AudioSegment
from scipy.ndimage import uniform_filter1d
import warnings
import logging
from sklearn.metrics import silhouette_score  # For dynamic speaker estimation

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Parameters with default values
MERGE_CONSECUTIVE_SEGMENTS = True
WINDOW_SIZE = 1.0          # in seconds
HOP_SIZE = 0.75            # in seconds
SMOOTHING_WINDOW_SIZE = 5
WHISPER_MODEL_NAME = "medium.en"  # Options: "tiny", "base", "small", "medium", "medium.en", "large", "large-v3"

def load_audio(filepath, target_sr=16000):
    # Load audio with librosa
    audio, sr = librosa.load(filepath, sr=target_sr)
    return audio, sr

def segment_audio(audio, sr, window_size=WINDOW_SIZE, hop_size=HOP_SIZE):
    # Split audio into overlapping windows
    window_length = int(window_size * sr)
    hop_length = int(hop_size * sr)
    segments = []
    timestamps = []
    for start in range(0, len(audio) - window_length + 1, hop_length):
        end = start + window_length
        segments.append(audio[start:end])
        timestamps.append((start / sr, end / sr))
    return segments, timestamps

def get_embeddings(segments, encoder):
    embeddings = []
    for seg in segments:
        emb = encoder.embed_utterance(seg)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    return embeddings

def smooth_labels(labels, window_size=SMOOTHING_WINDOW_SIZE):
    smoothed_labels = uniform_filter1d(labels.astype(float), size=window_size, mode='nearest')
    smoothed_labels = np.round(smoothed_labels).astype(int)
    return smoothed_labels

from sklearn.metrics import silhouette_score

def estimate_num_speakers(embeddings, min_speakers=1, max_speakers=10):
    """
    Estimate the optimal number of speakers using silhouette score.
    """
    best_score = -1
    best_num_speakers = min_speakers
    best_labels = np.zeros(len(embeddings))  # Default to one speaker

    # If only one speaker, just return one label for all
    if min_speakers == 1 and max_speakers == 1:
        logging.info(f"Detected one speaker, skipping clustering.")
        return best_labels

    for n_speakers in range(min_speakers, max_speakers + 1):
        # Perform clustering with a different number of speakers
        refinement_options = RefinementOptions(
            p_percentile=0.90,
            gaussian_blur_sigma=1,
        )
        clusterer = SpectralClusterer(
            min_clusters=n_speakers,
            max_clusters=n_speakers,
            refinement_options=refinement_options,
        )
        labels = clusterer.predict(embeddings)

        # Skip silhouette score if only 1 cluster
        if len(np.unique(labels)) == 1:
            logging.info(f"Only one cluster found with {n_speakers} speakers. Skipping silhouette score.")
            continue

        # Calculate the silhouette score for this clustering
        score = silhouette_score(embeddings, labels)

        logging.info(f"Number of speakers: {n_speakers}, Silhouette Score: {score}")

        # Keep track of the best score and corresponding number of speakers
        if score > best_score:
            best_score = score
            best_num_speakers = n_speakers
            best_labels = labels

    logging.info(f"Best number of speakers: {best_num_speakers} with a Silhouette Score of {best_score}")
    return best_labels

def transcribe_audio(filepath, model_name=WHISPER_MODEL_NAME):
    # Load Whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    # Transcribe audio
    result = model.transcribe(filepath)
    # Get segments
    segments = result["segments"]
    return segments

def assign_speakers_to_transcripts(transcript_segments, diarization_labels, diarization_timestamps):
    speaker_transcripts = []
    for segment in transcript_segments:
        start = segment['start']
        end = segment['end']
        text = segment['text']
        # Find the diarization label(s) for this segment
        speaker_label = 'Unknown'
        speaker_counts = {}
        for idx, (d_start, d_end) in enumerate(diarization_timestamps):
            # Check for overlap
            overlap = max(0, min(end, d_end) - max(start, d_start))
            if overlap > 0:
                label = f"Speaker {diarization_labels[idx]+1}"
                speaker_counts[label] = speaker_counts.get(label, 0) + overlap
        if speaker_counts:
            # Assign the speaker with the maximum overlap
            speaker_label = max(speaker_counts, key=speaker_counts.get)
        speaker_transcripts.append({
            'start': start,
            'end': end,
            'speaker': speaker_label,
            'text': text.strip()
        })
    return speaker_transcripts

def merge_consecutive_speaker_segments(speaker_transcripts):
    merged_segments = []
    current_segment = None

    for segment in speaker_transcripts:
        if current_segment is None:
            current_segment = segment.copy()
        elif segment['speaker'] == current_segment['speaker']:
            # Merge the text and update the end time
            current_segment['text'] += ' ' + segment['text']
            current_segment['end'] = segment['end']
        else:
            # Append the current segment and start a new one
            merged_segments.append(current_segment)
            current_segment = segment.copy()

    # Append the last segment
    if current_segment is not None:
        merged_segments.append(current_segment)

    return merged_segments

def format_output(speaker_transcripts):
    lines = []
    for segment in speaker_transcripts:
        start_time = format_timestamp(segment['start'])
        speaker = segment['speaker']
        text = segment['text']
        lines.append(f"=== {start_time} ({speaker}) ===\n{text}\n")
    return "\n".join(lines)

def format_timestamp(seconds):
    # Format seconds into hh:mm:ss format
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def main(audio_filepath, output_filepath=None):
    logging.info("Loading audio file...")
    audio, sr = load_audio(audio_filepath)
    logging.info("Segmenting audio...")
    segments, timestamps = segment_audio(audio, sr)
    logging.info("Computing embeddings...")
    encoder = VoiceEncoder()
    embeddings = get_embeddings(segments, encoder)
    logging.info("Estimating number of speakers and clustering embeddings...")
    labels = estimate_num_speakers(embeddings)  # Dynamic speaker estimation
    logging.info("Smoothing labels...")
    labels = smooth_labels(labels)
    logging.info("Transcribing audio...")
    transcript_segments = transcribe_audio(audio_filepath)
    logging.info("Assigning speaker labels to transcripts...")
    speaker_transcripts = assign_speakers_to_transcripts(transcript_segments, labels, timestamps)

    if MERGE_CONSECUTIVE_SEGMENTS:
        logging.info("Merging consecutive segments with the same speaker...")
        speaker_transcripts = merge_consecutive_speaker_segments(speaker_transcripts)

    output = format_output(speaker_transcripts)
    logging.info("=== Diarization and Transcription Output ===\n")
    print(output)

    if output_filepath:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        logging.info(f"\nTranscription has been saved to {output_filepath}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python diarization_script.py path_to_audio_file [output_textfile.txt]")
        sys.exit(1)
    audio_filepath = sys.argv[1]
    output_filepath = sys.argv[2] if len(sys.argv) > 2 else None
    main(audio_filepath, output_filepath)
