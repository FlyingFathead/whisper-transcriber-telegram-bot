# diarize_resemblyzer.py
# (From: https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/)
#
# == Local diarization (ECAPA or Resemblyzer) + Whisper transcription ==
#
# - Segments audio into overlapping windows (default 1.0s / 0.75s hop)
# - Optional WebRTC VAD gating (on by default; energy fallback if missing)
# - Embeds with:
#     * ECAPA-TDNN (speechbrain)  ← default, or
#     * Resemblyzer VoiceEncoder
# - Clusters via:
#     * GMM + BIC selection (default), or
#     * Spectral clustering + silhouette
# - Stronger separation defaults; optional pitch feature for M/F separation
# - Diagnostics for guard/silhouette/cluster sizes/centroid separation
# - Transcribes with Whisper and assigns speakers
# - Outputs .txt / .srt / .vtt
#
# Deps (pip):
#   numpy torch librosa spectralcluster scikit-learn whisper webrtcvad
#   speechbrain  (recommended; provides ECAPA-TDNN embedder)
#   resemblyzer (optional if using --embedder resemblyzer or as fallback)

import os, argparse, warnings, logging
import numpy as np
import torch
import librosa
import whisper
from spectralcluster import SpectralClusterer, RefinementOptions
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering

# --- Optional embedders ---
HAVE_ECAPA = False
try:
    from speechbrain.pretrained import EncoderClassifier
    HAVE_ECAPA = True
except Exception:
    HAVE_ECAPA = False

HAVE_RESEMBLYZER = False
try:
    from resemblyzer import VoiceEncoder
    HAVE_RESEMBLYZER = True
except Exception:
    HAVE_RESEMBLYZER = False

# Try webrtcvad; fall back to energy gate if missing.
try:
    import webrtcvad
    HAVE_WEBRTCVAD = True
except Exception:
    webrtcvad = None
    HAVE_WEBRTCVAD = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- Defaults (tuned to avoid collapsing to 1 spk) ----------
MERGE_CONSECUTIVE_SEGMENTS = True
WINDOW_SIZE = 0.8          # seconds (use >1.0 in faster interactions)
HOP_SIZE = 0.4             # seconds (0.75 default, use 0.3 for better turn detection)
SMOOTHING_WINDOW_SIZE = 7  # frames
DEFAULT_WHISPER_MODEL = "turbo"
VAD_FRAME_MS = 30
VAD_AGGR = 2
MIN_VOICED_RATIO = 0.2
DEFAULT_DEVICE = "auto"

# Less conservative than before:
GUARD_Q90 = 0.11           # if 90th pct cosine dist <= this → assume 1 spk; lower value = less likely to drop to 1
FLIMSY_SIL = 0.18          # silhouette below this → fallback agglomerative
FALLBACK_DIST = 0.15       # agglomerative distance threshold (cosine)
COLLAPSE_MAJ = 0.90        # collapse to 1 if largest cluster >= this fraction
COLLAPSE_SIL = 0.25        # collapse to 1 if final silhouette < this
COLLAPSE_CENT = 0.22       # collapse to 1 if min centroid distance <= this
MIN_CLUSTER_SIZE = 5       # clusters smaller than this are absorbed

# Changepoint recognition
CP_ENTER = 0.28
CP_EXIT  = 0.22
MIN_REGION_SEC = 1.2

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Device helpers ----------
def resolve_device(arg: str) -> str:
    arg = (arg or "auto").strip().lower()
    if arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda:0"
    if arg.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        try:
            idx = int(arg.split(":", 1)[1])
        except Exception:
            raise ValueError(f"Invalid CUDA device '{arg}'. Use 'cuda' or 'cuda:N'.")
        if idx < 0 or idx >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {arg} but only {torch.cuda.device_count()} CUDA device(s) are visible.")
        return f"cuda:{idx}"
    if arg == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device spec: {arg}")

# ---------- Audio utils ----------
def load_audio(filepath, target_sr=16000):
    audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
    if audio.size == 0:
        raise ValueError("Loaded empty audio.")
    audio = np.asarray(audio, dtype=np.float32)
    return np.clip(audio, -1.0, 1.0), sr

def segment_audio(audio, sr, window_size=WINDOW_SIZE, hop_size=HOP_SIZE):
    win = int(window_size * sr)
    hop = int(hop_size * sr)
    if win <= 0 or hop <= 0:
        raise ValueError("window_size and hop_size must be > 0.")
    if len(audio) < win:
        pad = win - len(audio)
        seg = np.pad(audio, (0, pad))
        return [seg], [(0.0, win / sr)]
    segments, timestamps = [], []
    for start in range(0, len(audio) - win + 1, hop):
        end = start + win
        segments.append(audio[start:end])
        timestamps.append((start / sr, end / sr))
    if (len(audio) % hop) > (0.5 * win) and (len(audio) - win) > 0:
        start = len(audio) - win
        end = len(audio)
        segments.append(audio[start:end])
        timestamps.append((start / sr, end / sr))
    return segments, timestamps

def _vad_mask_webrtc(audio, sr, frame_ms=VAD_FRAME_MS, aggressiveness=VAD_AGGR):
    vad = webrtcvad.Vad(int(aggressiveness))
    frame = int(sr * frame_ms / 1000)
    pad = (-len(audio)) % frame
    if pad:
        audio = np.pad(audio, (0, pad))
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    voiced = []
    step = frame * 2  # bytes
    for i in range(0, len(pcm16), step):
        chunk = pcm16[i: i + step]
        if len(chunk) < step:
            chunk = chunk + b"\x00" * (step - len(chunk))
        voiced.append(bool(vad.is_speech(chunk, sr)))
    return np.repeat(np.array(voiced, dtype=bool), frame)[: len(audio)]

def _vad_mask_energy(audio, sr, frame_ms=VAD_FRAME_MS, thresh_db=-45.0):
    frame = int(sr * frame_ms / 1000)
    pad = (-len(audio)) % frame
    if pad:
        audio = np.pad(audio, (0, pad))
    frames = audio.reshape(-1, frame)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > float(thresh_db)
    return np.repeat(voiced, frame)[: len(audio)]

def vad_mask(audio, sr, frame_ms=VAD_FRAME_MS, aggressiveness=VAD_AGGR):
    if HAVE_WEBRTCVAD:
        return _vad_mask_webrtc(audio, sr, frame_ms, aggressiveness)
    logging.warning("webrtcvad not found; using energy gate fallback.")
    return _vad_mask_energy(audio, sr, frame_ms)

def segment_audio_with_vad(
    audio,
    sr,
    window_size=WINDOW_SIZE,
    hop_size=HOP_SIZE,
    frame_ms=VAD_FRAME_MS,
    aggressiveness=VAD_AGGR,
    min_voiced_ratio=MIN_VOICED_RATIO,
):
    segments, timestamps = segment_audio(audio, sr, window_size, hop_size)
    mask = vad_mask(audio, sr, frame_ms=frame_ms, aggressiveness=aggressiveness)
    kept_segments, kept_timestamps = [], []
    for seg, (t0, t1) in zip(segments, timestamps):
        a0, a1 = int(t0 * sr), int(t1 * sr)
        a1 = min(a1, len(mask))
        if a1 <= a0:
            continue
        voiced_ratio = float(mask[a0:a1].mean())
        if voiced_ratio >= float(min_voiced_ratio):
            kept_segments.append(seg)
            kept_timestamps.append((t0, t1))
    return kept_segments, kept_timestamps

# ---------- Embeddings ----------
def _unit_norm(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def get_embeddings_resemblyzer(segments, encoder: "VoiceEncoder"):
    if not segments:
        return np.zeros((0, 256), dtype=np.float32)
    embs = []
    for seg in segments:
        emb = encoder.embed_utterance(seg)
        embs.append(emb.astype(np.float32))
    return np.vstack(embs)

def get_embeddings_ecapa(segments, sb_enc: "EncoderClassifier", dev: str):
    if not segments:
        return np.zeros((0, 192), dtype=np.float32)
    embs = []
    for seg in segments:
        wav = torch.from_numpy(seg).to(dev).float().unsqueeze(0)  # [1, T]
        with torch.no_grad():
            e = sb_enc.encode_batch(wav).squeeze().detach().cpu().numpy()
        embs.append(e.astype(np.float32))
    return np.vstack(embs)

# ---------- Pitch feature (optional: helps M/F separation) ----------
def _median_logf0(seg, sr):
    try:
        f0 = librosa.yin(seg, fmin=50, fmax=400, sr=sr)
        f0 = np.asarray(f0, dtype=np.float32)
        f0[~np.isfinite(f0)] = np.nan
        m = np.nanmedian(f0)
        if not np.isfinite(m) or m <= 0:
            return np.nan
        return float(np.log(m))
    except Exception:
        return np.nan

def compute_pitch_feature_per_window(segments, sr):
    if not segments:
        return np.zeros((0,), dtype=np.float32)
    vals = np.array([_median_logf0(seg, sr) for seg in segments], dtype=np.float32)
    good = np.isfinite(vals)
    if good.any():
        mu = float(np.nanmean(vals[good]))
        sd = float(np.nanstd(vals[good]) + 1e-8)
        z = (vals - mu) / sd
        z[~np.isfinite(z)] = 0.0
        return z.astype(np.float32)
    return np.zeros_like(vals, dtype=np.float32)

# ---------- Label smoothing ----------
def smooth_labels(labels, window_size=SMOOTHING_WINDOW_SIZE):
    if window_size is None or window_size <= 1 or labels.size == 0:
        return labels.astype(int)
    arr = uniform_filter1d(labels.astype(float), size=int(window_size), mode="nearest")
    return np.round(arr).astype(int)

# --- replace enforce_min_run with a small-N safe version ---
def enforce_min_run(labels: np.ndarray, min_run: int = 3) -> np.ndarray:
    if labels.size == 0 or min_run <= 1:
        return labels
    # Only enforce if we have enough labels to form alternating runs
    if labels.size < 2 * min_run:
        return labels
    y = labels.copy()
    i = 0
    while i < len(y):
        j = i + 1
        while j < len(y) and y[j] == y[i]:
            j += 1
        if (j - i) < min_run:
            if i == 0:
                y[i:j] = y[j] if j < len(y) else y[i]
            else:
                y[i:j] = y[i - 1]
        i = j
    return y

# --- Changepoint recognition helpers ---
def _regionize_by_changepoints(X_win, ts_win, enter=CP_ENTER, exit=CP_EXIT, min_region_sec=MIN_REGION_SEC):
    X = _unit_norm(np.asarray(X_win, dtype=np.float32))
    N = len(X)
    if N == 0:
        return [], []
    if N == 1:
        return [(0, 0)], [ts_win[0]]
    d = np.maximum(0.0, (cosine_distances(X[:-1], X[1:]).diagonal()))
    on = False
    cuts = [0]
    for i, val in enumerate(d, start=1):
        if not on and val >= float(enter):
            on = True; cuts.append(i)
        elif on and val <= float(exit):
            on = False; cuts.append(i)
    if cuts[-1] != N:
        cuts.append(N)
    regions, ts_regions = [], []
    cur_s = cuts[0]
    for cur_e in cuts[1:]:
        s, e = cur_s, cur_e - 1
        t0, _ = ts_win[s]
        _, t1 = ts_win[e]
        if (t1 - t0) >= float(min_region_sec):
            regions.append((s, e)); ts_regions.append((t0, t1)); cur_s = cur_e
    if cur_s < N:
        s, e = cur_s, N - 1
        t0, _ = ts_win[s]; _, t1 = ts_win[e]
        if regions and (t1 - t0) < float(min_region_sec):
            ps, pe = regions[-1]; regions[-1] = (ps, e)
            pt0, _ = ts_regions[-1]; ts_regions[-1] = (pt0, t1)
        else:
            regions.append((s, e)); ts_regions.append((t0, t1))
    if not regions:
        regions = [(0, N - 1)]; ts_regions = [(ts_win[0][0], ts_win[-1][1])]
    return regions, ts_regions

def _mean_embs_by_regions(X_win, regions):
    if not regions:
        return np.zeros((0, X_win.shape[1]), dtype=np.float32)
    means = []
    for s, e in regions:
        means.append(X_win[s:e+1].mean(axis=0))
    return _unit_norm(np.vstack(means).astype(np.float32))

def _expand_region_labels_to_windows(regions, lab_regions, N):
    y = np.zeros(N, dtype=int)
    for (s, e), lab in zip(regions, lab_regions):
        y[s:e+1] = int(lab)
    return y

# ---------- Clustering helpers ----------
def single_speaker_guard(embeddings, guard_q90=GUARD_Q90):
    X = _unit_norm(np.asarray(embeddings, dtype=np.float32))
    n = len(X)
    if n < 4:
        return True
    D = cosine_distances(X)
    tri = D[np.triu_indices_from(D, k=1)]
    if tri.size == 0:
        return True
    q90 = float(np.quantile(tri, 0.90))
    return q90 <= float(guard_q90)

def estimate_num_speakers_silhouette(embeddings, min_speakers=1, max_speakers=10):
    X = _unit_norm(np.asarray(embeddings, dtype=np.float32))
    n = len(X)
    if n < 3:
        return np.zeros(n, dtype=int)
    best_score = -1.0
    best_labels = np.zeros(n, dtype=int)
    refinement_options = RefinementOptions(p_percentile=0.90, gaussian_blur_sigma=1)
    for k in range(int(min_speakers), int(max_speakers) + 1):
        clusterer = SpectralClusterer(min_clusters=k, max_clusters=k, refinement_options=refinement_options)
        labels = clusterer.predict(X)
        if len(np.unique(labels)) < 2:
            continue
        try:
            score = float(silhouette_score(X, labels))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_labels = labels
    return best_labels

def cluster_gmm_bic(embeddings, k_min=1, k_max=12, covariance_type="diag"):
    X = _unit_norm(np.asarray(embeddings, dtype=np.float32))
    n = len(X)
    if n <= 1:
        return np.zeros(n, dtype=int)

    # can't have more clusters than points
    k_max = int(min(k_max, n))
    k_min = int(min(max(1, k_min), k_max))

    # Small-N fallback: avoid "always 1 cluster" when regionization yields few points
    if n < 8:
        # If user asked for >=2 speakers, try to honor it
        k = max(2, k_min) if k_max >= 2 else 1
        if k == 1:
            return np.zeros(n, dtype=int)
        try:
            ac = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        except TypeError:  # older sklearn
            ac = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
        return ac.fit_predict(X)

    # Normal BIC selection
    bics, gmms = [], []
    for k in range(k_min, k_max + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=2, random_state=0)
            gmm.fit(X)
            bics.append(gmm.bic(X)); gmms.append(gmm)
        except Exception:
            continue
    if not bics:
        return np.zeros(n, dtype=int)
    best = int(np.argmin(bics))
    return gmms[best].predict(X)

def _centroids(X, labels):
    labs = np.unique(labels)
    C = []
    for l in labs:
        C.append(X[labels == l].mean(axis=0))
    C = _unit_norm(np.vstack(C))
    return labs, C

def _absorb_tiny_clusters(X, labels, min_cluster_size=MIN_CLUSTER_SIZE):
    labs, C = _centroids(X, labels)
    sizes = {l: int((labels == l).sum()) for l in labs}
    big = [l for l in labs if sizes[l] >= min_cluster_size]
    tiny = [l for l in labs if sizes[l] <  min_cluster_size]
    if not tiny or not big:
        return labels
    C_big = _unit_norm(np.vstack([C[list(labs).index(l)] for l in big]))
    new = labels.copy()
    for t in tiny:
        idx = np.where(labels == t)[0]
        ct = _unit_norm(C[list(labs).index(t)][None, :])
        d = cosine_distances(ct, C_big)[0]
        target = big[int(np.argmin(d))]
        new[idx] = target
    uniq = np.unique(new)
    remap = {l:i for i,l in enumerate(uniq)}
    return np.array([remap[l] for l in new], dtype=int)

def _collapse_to_one(X, labels,
                     collapse_majority=COLLAPSE_MAJ,
                     collapse_sil=COLLAPSE_SIL,
                     collapse_centroid=COLLAPSE_CENT):
    n = len(labels)
    if n == 0 or len(np.unique(labels)) <= 1:
        return np.zeros(n, dtype=int)
    _, counts = np.unique(labels, return_counts=True)
    if (counts.max() / float(n)) >= float(collapse_majority):
        return np.zeros(n, dtype=int)
    try:
        sil = float(silhouette_score(X, labels))
    except Exception:
        sil = -1.0
    if sil < float(collapse_sil):
        return np.zeros(n, dtype=int)
    labs, C = _centroids(X, labels)
    if len(C) >= 2:
        D = cosine_distances(C)
        dm = D[np.triu_indices_from(D, k=1)].min()
        if dm <= float(collapse_centroid):
            return np.zeros(n, dtype=int)
    return None

def _describe_partition(X, labels, prefix=""):
    ncl = len(np.unique(labels))
    try:
        sil = float(silhouette_score(X, labels)) if ncl > 1 else -1.0
    except Exception:
        sil = -1.0
    msg = f"{prefix} clusters={ncl}, silhouette={sil:.3f}"
    if ncl > 1:
        _, counts = np.unique(labels, return_counts=True)
        labs, C = _centroids(X, labels)
        D = cosine_distances(C)
        dm = D[np.triu_indices_from(D, k=1)].min()
        msg += f", sizes={list(counts)}, min_centroid_dist={dm:.3f}"
    logging.info(msg)

def pick_labels(embeddings, method="bic", min_speakers=1, max_speakers=4,
                force_n=None, fallback_dist=FALLBACK_DIST, flimsy_sil=FLIMSY_SIL,
                guard_q90=GUARD_Q90, collapse_majority=COLLAPSE_MAJ, collapse_sil=COLLAPSE_SIL,
                collapse_centroid=COLLAPSE_CENT, min_cluster_size=MIN_CLUSTER_SIZE,
                no_collapse=False, no_guard=False):
    """
    Returns raw labels for provided feature vectors (no temporal postproc).
    If force_n is set, it is a HARD constraint (no guard, no flimsy-fallback, no collapse).
    """
    X = _unit_norm(np.asarray(embeddings, dtype=np.float32))
    n = len(X)
    k_req = int(max(1, force_n))
    k = int(min(k_req, max_speakers, n))
    if k < k_req:
        logging.warning("force_n=%d but only %d sample(s); using k=%d", k_req, n, k)
    if k == 1:
        return np.zeros(n, dtype=int)
    if n == 0:
        return np.zeros(0, dtype=int)

    # ---- HARD K MODE ----
    if force_n is not None and force_n >= 1:
        k = int(max(1, min(force_n, max_speakers)))
        logging.info("HARD-K mode: force_n=%d (bypass guard/fallback/collapse)", k)
        # Try Spectral with fixed K first
        clusterer = SpectralClusterer(
            min_clusters=k, max_clusters=k,
            refinement_options=RefinementOptions(p_percentile=0.90, gaussian_blur_sigma=1)
        )
        try:
            labels = clusterer.predict(X)
        except Exception:
            labels = np.zeros(n, dtype=int)

        # If degenerate (<K uniques), use Agglomerative(K) with tiny jitter to break ties
        if len(np.unique(labels)) < k:
            logging.info("Spectral returned %d<%d clusters → fallback Agglomerative(K)",
                         len(np.unique(labels)), k)
            Xj = (X + 1e-4 * np.random.randn(*X.shape)).astype(np.float32)
            try:
                ac = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
            except TypeError:
                ac = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
            labels = ac.fit_predict(Xj)

        _describe_partition(X, labels, prefix="after HARD-K")
        return labels

    # ---- NORMAL MODE (no force_n) ----

    # 0) single-speaker guard
    D = cosine_distances(X)
    tri = D[np.triu_indices_from(D, k=1)]
    q90 = float(np.quantile(tri, 0.90)) if tri.size else 0.0
    logging.info("guard check: q90=%.3f (thresh=%.3f) %s", q90, guard_q90,
                 "(DISABLED)" if no_guard else "")
    if not no_guard and q90 <= float(guard_q90):
        logging.info("→ guard fired: returning 1 cluster")
        return np.zeros(n, dtype=int)

    # 1) initial clustering
    if method == "bic":
        labels = cluster_gmm_bic(X, k_min=min_speakers, k_max=max_speakers)
    elif method == "silhouette":
        labels = estimate_num_speakers_silhouette(X, min_speakers, max_speakers)
    else:
        raise ValueError(f"Unknown method: {method}")

    _describe_partition(X, labels, prefix="after primary")

    # 2) flimsy? try bounded Agglomerative(K) — NOT distance_threshold
    if len(np.unique(labels)) > 1:
        try:
            sil0 = float(silhouette_score(X, labels))
        except Exception:
            sil0 = -1.0
        if sil0 < float(flimsy_sil):
            k0 = max(int(min_speakers), min(int(max_speakers), len(np.unique(labels))))
            k0 = max(2, k0)
            logging.info("silhouette=%.3f < %.3f → fallback Agglomerative(n_clusters=%d)",
                         sil0, flimsy_sil, k0)
            try:
                ac = AgglomerativeClustering(n_clusters=k0, metric="cosine", linkage="average")
            except TypeError:
                ac = AgglomerativeClustering(n_clusters=k0, affinity="cosine", linkage="average")
            labels = ac.fit_predict(X)
            _describe_partition(X, labels, prefix="after fallback")

    # 3) absorb tiny clusters
    if len(np.unique(labels)) > 1:
        labels = _absorb_tiny_clusters(X, labels, min_cluster_size=min_cluster_size)
        _describe_partition(X, labels, prefix="after absorb")

    # 4) optional collapse-to-1
    if not no_collapse and len(np.unique(labels)) > 1:
        maybe = _collapse_to_one(
            X, labels,
            collapse_majority=collapse_majority,
            collapse_sil=collapse_sil,
            collapse_centroid=collapse_centroid,
        )
        if maybe is not None:
            logging.info("final collapse triggered → 1 cluster")
            return maybe

    return labels

# ---------- Transcription ----------
def transcribe_audio(filepath, model_name=DEFAULT_WHISPER_MODEL, language=None, device="cpu"):
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(filepath, language=language)
    return result.get("segments", [])

# ---------- Alignment + formatting ----------
def assign_speakers_to_transcripts(transcript_segments, diar_labels, diar_timestamps):
    if len(diar_labels) != len(diar_timestamps):
        if len(diar_labels) == 0 or len(diar_timestamps) == 0:
            out = []
            for seg in transcript_segments:
                out.append(
                    {
                        "start": float(seg.get("start", 0.0)),
                        "end": float(seg.get("end", 0.0)),
                        "speaker": "Speaker 1",
                        "text": (seg.get("text") or "").strip(),
                    }
                )
            return out
        raise ValueError("Mismatch between diarization labels and timestamps.")

    speaker_transcripts = []
    for seg in transcript_segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = (seg.get("text") or "").strip()
        counts = {}
        for idx, (d0, d1) in enumerate(diar_timestamps):
            overlap = max(0.0, min(end, d1) - max(start, d0))
            if overlap > 0.0:
                label = f"Speaker {int(diar_labels[idx]) + 1}"
                counts[label] = counts.get(label, 0.0) + overlap
        spk = max(counts, key=counts.get) if counts else "Speaker 1"
        speaker_transcripts.append({"start": start, "end": end, "speaker": spk, "text": text})
    return speaker_transcripts

def merge_consecutive_speaker_segments(speaker_transcripts, max_gap=0.2):
    if not speaker_transcripts:
        return []
    merged = []
    cur = speaker_transcripts[0].copy()
    for seg in speaker_transcripts[1:]:
        if seg["speaker"] == cur["speaker"] and (seg["start"] - cur["end"]) <= float(max_gap):
            cur["text"] = (cur["text"] + " " + seg["text"]).strip()
            cur["end"] = seg["end"]
        else:
            merged.append(cur)
            cur = seg.copy()
    merged.append(cur)
    return merged

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def format_output_text(speaker_transcripts, include_end=True):
    lines = []
    for seg in speaker_transcripts:
        if include_end:
            head = f"=== {format_timestamp(seg['start'])}-{format_timestamp(seg['end'])} ({seg['speaker']}) ==="
        else:
            head = f"=== {format_timestamp(seg['start'])} ({seg['speaker']}) ==="
        lines.append(f"{head}\n{seg['text']}\n")
    return "\n".join(lines)

def _fmt_srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _fmt_vtt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def write_srt(speaker_transcripts, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(speaker_transcripts, 1):
            f.write(
                f"{i}\n{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}\n"
                f"({seg['speaker']}) {seg['text'].strip()}\n\n"
            )

def write_vtt(speaker_transcripts, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in speaker_transcripts:
            f.write(
                f"{_fmt_vtt_time(seg['start'])} --> {_fmt_vtt_time(seg['end'])}\n"
                f"({seg['speaker']}) {seg['text'].strip()}\n\n"
            )

# ---------- Main ----------
def main(
    audio_filepath,
    output_filepath=None,
    method="bic",
    force_n=None,
    whisper_model=DEFAULT_WHISPER_MODEL,
    language=None,
    device=DEFAULT_DEVICE,
    use_vad=True,
    window_size=WINDOW_SIZE,
    hop_size=HOP_SIZE,
    vad_frame_ms=VAD_FRAME_MS,
    vad_aggr=VAD_AGGR,
    min_voiced_ratio=MIN_VOICED_RATIO,
    smoothing_window=SMOOTHING_WINDOW_SIZE,
    min_run=3,
    merge_consecutive=MERGE_CONSECUTIVE_SEGMENTS,
    max_gap_merge=0.2,
    min_speakers=1,
    max_speakers=4,
    single_guard=GUARD_Q90,
    fallback_dist=FALLBACK_DIST,
    flimsy_sil=FLIMSY_SIL,
    collapse_majority=COLLAPSE_MAJ,
    collapse_sil=COLLAPSE_SIL,
    collapse_centroid=COLLAPSE_CENT,
    seed=1337,
    include_span=True,
    embedder="ecapa",
    use_cp=True,
    no_collapse=True,      # default: DO NOT collapse to 1
    no_guard=False,        # allow disabling the "single-speaker guard"
    use_pitch=False,       # optional pitch feature
):
    # Resolve device
    dev = resolve_device(device)
    if dev.startswith("cuda:"):
        try:
            torch.cuda.set_device(int(dev.split(":", 1)[1]))
        except Exception:
            pass

    # Repro-ish
    try:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass

    logging.info("Loading audio...")
    audio, sr = load_audio(audio_filepath)

    logging.info("Segmenting audio%s...", " with VAD" if use_vad else "")
    if use_vad:
        segments, timestamps = segment_audio_with_vad(
            audio, sr,
            window_size=window_size,
            hop_size=hop_size,
            frame_ms=vad_frame_ms,
            aggressiveness=vad_aggr,
            min_voiced_ratio=min_voiced_ratio,
        )
        if not segments:
            logging.warning("VAD removed all windows; falling back to ungated segmentation.")
            segments, timestamps = segment_audio(audio, sr, window_size, hop_size)
    else:
        segments, timestamps = segment_audio(audio, sr, window_size, hop_size)

    # ----- Embeddings (ECAPA default with fallback) -----
    if embedder == "ecapa":
        if HAVE_ECAPA:
            logging.info("Embedding with ECAPA-TDNN (speechbrain) on %s ...", dev)
            sb_enc = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": dev}
            )
            win_embs = get_embeddings_ecapa(segments, sb_enc, dev)
        else:
            if not HAVE_RESEMBLYZER:
                raise RuntimeError("ECAPA requested but speechbrain not available, and Resemblyzer not installed.")
            logging.warning("speechbrain not found; falling back to Resemblyzer VoiceEncoder.")
            encoder = VoiceEncoder(device=torch.device(dev))
            win_embs = get_embeddings_resemblyzer(segments, encoder)
    else:
        if not HAVE_RESEMBLYZER:
            raise RuntimeError("Resemblyzer requested but not installed.")
        logging.info("Embedding with Resemblyzer VoiceEncoder on %s ...", dev)
        encoder = VoiceEncoder(device=torch.device(dev))
        win_embs = get_embeddings_resemblyzer(segments, encoder)

    X_win = _unit_norm(win_embs)

    # Optional pitch feature per window
    pitch_win = None
    if use_pitch:
        logging.info("Computing pitch feature (log-F0 z-score) per window ...")
        pitch_win = compute_pitch_feature_per_window(segments, sr).reshape(-1, 1)

    # --- CHANGE-POINT REGIONIZATION ---
    if use_cp:
        regions, ts_regions = _regionize_by_changepoints(
            X_win, timestamps, enter=CP_ENTER, exit=CP_EXIT, min_region_sec=MIN_REGION_SEC
        )
        X_reg = _mean_embs_by_regions(X_win, regions)
        # average pitch over region, if used
        if use_pitch and pitch_win is not None:
            pr = []
            for s, e in regions:
                pr.append(float(np.mean(pitch_win[s:e+1])))
            pitch_reg = np.array(pr, dtype=np.float32).reshape(-1, 1)
            feat_for_cluster = np.hstack([X_reg, pitch_reg])
        else:
            feat_for_cluster = X_reg
        logging.info("Regionized %d windows -> %d regions", len(X_win), len(regions))
        diag_D = cosine_distances(feat_for_cluster)
        tri = diag_D[np.triu_indices_from(diag_D, k=1)]
        q90 = float(np.quantile(tri, 0.90)) if tri.size else 0.0
        logging.info("q90 distance after regionization/features: %.3f", q90)
    else:
        regions = [(i, i) for i in range(len(X_win))]
        ts_regions = timestamps
        if use_pitch and pitch_win is not None:
            feat_for_cluster = np.hstack([X_win, pitch_win])
        else:
            feat_for_cluster = X_win

    # --- after computing regions / X_reg / feat_for_cluster in the CP branch ---
    # If user demands multi-speaker but CP produced <2 regions, fall back to window-level features
    if force_n and int(force_n) >= 2 and feat_for_cluster.shape[0] < 2:
        logging.warning(
            "CP produced %d region(s); falling back to window-level clustering.",
            feat_for_cluster.shape[0]
        )
        regions = [(i, i) for i in range(len(X_win))]
        ts_regions = timestamps
        feat_for_cluster = (
            np.hstack([X_win, pitch_win]) if (use_pitch and pitch_win is not None) else X_win
        )

    logging.info("Clustering (method=%s%s)...", method, f", force_n={force_n}" if force_n else "")
    lab_regions = pick_labels(
        feat_for_cluster,
        method=method,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        force_n=force_n,
        fallback_dist=fallback_dist,
        flimsy_sil=flimsy_sil,
        guard_q90=single_guard,
        collapse_majority=collapse_majority,
        collapse_sil=collapse_sil,
        collapse_centroid=collapse_centroid,
        min_cluster_size=MIN_CLUSTER_SIZE,
        no_collapse=no_collapse,
        no_guard=no_guard,
    )

    # Temporal post-processing on REGION labels
    labels = lab_regions
    if labels.size and smoothing_window and smoothing_window > 1:
        labels = smooth_labels(labels, smoothing_window)
    if labels.size and min_run and min_run > 1:
        labels = enforce_min_run(labels, min_run=min_run)

    logging.info("Transcribing with Whisper (%s) on %s...", whisper_model, dev)
    transcript_segments = transcribe_audio(audio_filepath, model_name=whisper_model, language=language, device=dev)

    diar_ts = ts_regions
    speaker_transcripts = assign_speakers_to_transcripts(transcript_segments, labels, diar_ts)

    if merge_consecutive:
        logging.info("Merging consecutive segments by same speaker (max_gap=%.2fs)...", max_gap_merge)
        speaker_transcripts = merge_consecutive_speaker_segments(speaker_transcripts, max_gap=max_gap_merge)

    txt = format_output_text(speaker_transcripts, include_end=include_span)

    if not output_filepath:
        print(txt)
    else:
        low = output_filepath.lower()
        if low.endswith(".srt"):
            write_srt(speaker_transcripts, output_filepath)
        elif low.endswith(".vtt"):
            write_vtt(speaker_transcripts, output_filepath)
        else:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(txt)
        logging.info("Saved to %s", output_filepath)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Local diarization (ECAPA / Resemblyzer) + Whisper transcription")
    ap.add_argument("audio", help="Path to audio file")
    ap.add_argument("-o", "--out", dest="out", default=None, help="Output (.txt | .srt | .vtt). Default: stdout")

    # embedder backend
    ap.add_argument("--embedder", choices=["ecapa", "resemblyzer"], default="ecapa",
                    help="Embedding backend (default: ecapa). Falls back to Resemblyzer if ECAPA unavailable.")

    # simple knobs
    ap.add_argument("--device", default=DEFAULT_DEVICE, help="cpu | cuda | cuda:N | auto (default)")
    ap.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL, help="Whisper model name")
    ap.add_argument("--language", default=None, help="Force language for Whisper (e.g., fi, en)")
    ap.add_argument("--force-n", type=int, default=None, help="Force exact number of speakers")

    # clustering
    ap.add_argument("--method", default="bic", choices=["bic", "silhouette"], help="Speaker clustering method")
    ap.add_argument("--min-speakers", type=int, default=1, help="Lower bound for discovery")
    ap.add_argument("--max-speakers", type=int, default=4, help="Upper bound for discovery")

    # VAD + windows
    ap.add_argument("--no-vad", action="store_true", help="Disable WebRTC VAD gating")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--span", dest="span", action="store_true",
                       help="Include end time in .txt headers (default)")
    group.add_argument("--no-span", dest="span", action="store_false",
                       help="Hide end time in .txt headers")
    ap.set_defaults(span=True)

    ap.add_argument("--window", type=float, default=WINDOW_SIZE, help="Window size (sec)")
    ap.add_argument("--hop", type=float, default=HOP_SIZE, help="Hop size (sec)")
    ap.add_argument("--vad-frame-ms", type=int, default=VAD_FRAME_MS, help="VAD frame size (10/20/30 ms)")
    ap.add_argument("--vad-aggr", type=int, default=VAD_AGGR, help="VAD aggressiveness 0..3")
    ap.add_argument("--min-voiced", type=float, default=MIN_VOICED_RATIO, help="Min voiced ratio per window to keep")

    # postproc + merging
    ap.add_argument("--smooth", type=int, default=SMOOTHING_WINDOW_SIZE, help="Label smoothing window (frames)")
    ap.add_argument("--min-run", type=int, default=3, help="Minimum consecutive windows per speaker after smoothing")
    ap.add_argument("--no-merge", action="store_true", help="Do not merge consecutive segments")
    ap.add_argument("--merge-gap", type=float, default=0.2, help="Max gap (sec) to merge same-speaker segments")

    # toggles
    ap.add_argument("--no-cp", action="store_true", help="Disable change-point regionization")
    ap.add_argument("--no-collapse", action="store_true", help="Disable final collapse-to-one heuristic (default on)")
    ap.add_argument("--no-guard", action="store_true", help="Disable single-speaker guard")
    ap.add_argument("--pitch", action="store_true", help="Append z-scored log-F0 to features")

    # thresholds
    ap.add_argument("--single-guard", type=float, default=GUARD_Q90, help="Guard: assume 1 spk if q90 dist <= this")
    ap.add_argument("--flimsy-sil", type=float, default=FLIMSY_SIL, help="Silhouette below this triggers fallback")
    ap.add_argument("--fallback-dist", type=float, default=FALLBACK_DIST, help="Agglomerative distance threshold")
    ap.add_argument("--collapse-majority", type=float, default=COLLAPSE_MAJ, help="Collapse to 1 if largest cluster ≥ this")
    ap.add_argument("--collapse-sil", type=float, default=COLLAPSE_SIL, help="Collapse to 1 if silhouette < this")
    ap.add_argument("--collapse-centroid", type=float, default=COLLAPSE_CENT, help="Collapse to 1 if min centroid dist ≤ this")

    # misc
    ap.add_argument("--seed", type=int, default=1337)

    ap.set_defaults(no_collapse=True)  # default: keep discovered clusters

    args = ap.parse_args()

    main(
        audio_filepath=args.audio,
        output_filepath=args.out,
        method=args.method,
        force_n=args.force_n,
        whisper_model=args.whisper_model,
        language=args.language,
        device=args.device,
        use_vad=(not args.no_vad),
        window_size=args.window,
        hop_size=args.hop,
        vad_frame_ms=args.vad_frame_ms,
        vad_aggr=args.vad_aggr,
        min_voiced_ratio=args.min_voiced,
        smoothing_window=args.smooth,
        min_run=args.min_run,
        merge_consecutive=(not args.no_merge),
        max_gap_merge=args.merge_gap,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        single_guard=args.single_guard,
        fallback_dist=args.fallback_dist,
        flimsy_sil=args.flimsy_sil,
        collapse_majority=args.collapse_majority,
        collapse_sil=args.collapse_sil,
        collapse_centroid=args.collapse_centroid,
        seed=args.seed,
        include_span=args.span,
        embedder=args.embedder,
        use_cp=(not args.no_cp),
        no_collapse=args.no_collapse,
        no_guard=args.no_guard,
        use_pitch=args.pitch,
    )

# # /// ALT: defaults to 1; conservative. 
# # /// GMM+BIC => sanity-checked split silhouette => min cluster size => centroid separateion
# # /// uncomment + copy-paste in-place above to see if it works better in your use case scenario.

# from sklearn.metrics.pairwise import cosine_distances

# def _unit_norm(X: np.ndarray) -> np.ndarray:
#     X = np.asarray(X, dtype=np.float32)
#     n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
#     return X / n

# def _centroid_separation(X, labels):
#     labs = np.unique(labels)
#     if len(labs) < 2:
#         return 1.0
#     cents = []
#     for l in labs:
#         cents.append(X[labels == l].mean(axis=0, keepdims=True))
#     C = np.vstack(cents)
#     D = cosine_distances(_unit_norm(C))
#     tri = D[np.triu_indices_from(D, k=1)]
#     return float(tri.min()) if tri.size else 1.0

# def _min_cluster_prop(labels):
#     vals, counts = np.unique(labels, return_counts=True)
#     return float(counts.min()) / float(labels.size)

# def single_speaker_guard(embeddings, guard_q90=0.18):
#     """
#     If the 90th percentile pairwise cosine distance is small, assume single speaker.
#     guard_q90 is *adaptive* if not given (we bump it slightly when audio is short/noisy).
#     """
#     X = _unit_norm(np.asarray(embeddings, dtype=np.float32))
#     n = len(X)
#     if n < 4:
#         return True
#     D = cosine_distances(X)
#     tri = D[np.triu_indices_from(D, k=1)]
#     if tri.size == 0:
#         return True
#     q90 = float(np.quantile(tri, 0.90))
#     return q90 <= float(guard_q90)

# def estimate_num_speakers_silhouette(embeddings, min_speakers=1, max_speakers=10):
#     X = _unit_norm(np.asarray(embeddings, dtype=np.float32))
#     n = len(X)
#     if n < 3:
#         return np.zeros(n, dtype=int)
#     best_score = -1.0
#     best_labels = np.zeros(n, dtype=int)
#     refinement_options = RefinementOptions(p_percentile=0.90, gaussian_blur_sigma=1)
#     for k in range(int(min_speakers), int(max_speakers) + 1):
#         clusterer = SpectralClusterer(min_clusters=k, max_clusters=k, refinement_options=refinement_options)
#         labels = clusterer.predict(X)
#         if len(np.unique(labels)) < 2:
#             continue
#         try:
#             score = float(silhouette_score(X, labels))
#         except Exception:
#             continue
#         if score > best_score:
#             best_score = score
#             best_labels = labels
#     return best_labels

# def cluster_gmm_bic(embeddings, k_min=1, k_max=6, covariance_type="diag"):
#     X = _unit_norm(np.asarray(embeddings, dtype=np.float32))
#     n = len(X)
#     if n < 8:
#         return np.zeros(n, dtype=int)
#     bics, gmms = [], []
#     for k in range(int(k_min), int(k_max) + 1):
#         try:
#             gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=2, random_state=0)
#             gmm.fit(X)
#             bics.append(gmm.bic(X)); gmms.append(gmm)
#         except Exception:
#             continue
#     if not bics:
#         return np.zeros(n, dtype=int)
#     best = int(np.argmin(bics))
#     return gmms[best].predict(X)

# def _merge_closest_centroids(X, labels, min_sep=0.12):
#     """
#     If two clusters are essentially the same (centroid cosine distance < min_sep),
#     merge them once.
#     """
#     labs = np.unique(labels)
#     if len(labs) < 2:
#         return labels
#     # compute centroids
#     cents = {l: X[labels == l].mean(axis=0, keepdims=True) for l in labs}
#     order = sorted(labs)
#     C = np.vstack([cents[l] for l in order])
#     D = cosine_distances(_unit_norm(C))
#     np.fill_diagonal(D, 1.0)
#     i, j = divmod(D.argmin(), D.shape[1])
#     if D[i, j] >= float(min_sep):
#         return labels
#     a, b = order[i], order[j]
#     new = labels.copy()
#     new[new == b] = a
#     # reindex to 0..K-1 for cleanliness
#     uniq = np.unique(new)
#     remap = {lab: k for k, lab in enumerate(uniq)}
#     return np.vectorize(remap.get)(new)

# def pick_labels(
#     embeddings,
#     method="bic",
#     min_speakers=1,
#     max_speakers=4,
#     force_n=None,
#     flimsy_sil=0.18,          # slightly stricter so we default to 1 unless clear split
#     min_prop=0.10,            # smallest cluster must have at least 10% of windows
#     min_centroid_sep=0.14,    # cosine distance between cluster centroids
#     guard_q90=0.20,           # nudge guard up a hair to avoid oversplitting solos
# ):
#     """
#     Autopilot: prefer 1 speaker unless evidence for >1 is strong on multiple tests.
#     """
#     X = _unit_norm(np.asarray(embeddings, dtype=np.float32))
#     n = len(X)
#     if n == 0:
#         return np.zeros(0, dtype=int)

#     # 0) hard clamp
#     if force_n is not None and force_n >= 1:
#         k = int(max(1, min(force_n, max_speakers)))
#         clusterer = SpectralClusterer(
#             min_clusters=k, max_clusters=k,
#             refinement_options=RefinementOptions(p_percentile=0.90, gaussian_blur_sigma=1)
#         )
#         return clusterer.predict(X)

#     # 1) strong single-speaker bias
#     if single_speaker_guard(X, guard_q90=guard_q90):
#         return np.zeros(n, dtype=int)

#     # 2) primary clustering
#     if method == "bic":
#         labels = cluster_gmm_bic(X, k_min=min_speakers, k_max=max_speakers)
#     elif method == "silhouette":
#         labels = estimate_num_speakers_silhouette(X, min_speakers, max_speakers)
#     else:
#         raise ValueError(f"Unknown method: {method}")

#     # 3) sanity checks
#     if len(np.unique(labels)) < 2:
#         return np.zeros(n, dtype=int)

#     try:
#         sil = float(silhouette_score(X, labels))
#     except Exception:
#         sil = -1.0
#     prop = _min_cluster_prop(labels)
#     sep = _centroid_separation(X, labels)

#     if sil < flimsy_sil or prop < min_prop or sep < min_centroid_sep:
#         # try one centroid-merge to salvage clean 2-cluster cases
#         merged = _merge_closest_centroids(X, labels, min_sep=min_centroid_sep)
#         if len(np.unique(merged)) >= 2:
#             try:
#                 sil2 = float(silhouette_score(X, merged))
#             except Exception:
#                 sil2 = -1.0
#             prop2 = _min_cluster_prop(merged)
#             sep2 = _centroid_separation(X, merged)
#             if sil2 >= flimsy_sil and prop2 >= min_prop and sep2 >= min_centroid_sep:
#                 labels = merged
#             else:
#                 return np.zeros(n, dtype=int)
#         else:
#             return np.zeros(n, dtype=int)

#     return labels
# # // EOF //

# ===========================================================================

# # /// OLD VERSION ///
# # diarize_resemblyzer.py
# # (From: https://github.com/FlyingFathead/whisper-transcriber-telegram-bot/)

# import os, argparse
# import numpy as np
# import torch
# from resemblyzer import VoiceEncoder, preprocess_wav
# from spectralcluster import SpectralClusterer, RefinementOptions
# import librosa
# import whisper
# from pydub import AudioSegment
# from scipy.ndimage import uniform_filter1d
# import warnings
# import logging
# from sklearn.metrics import silhouette_score  # For dynamic speaker estimation

# # To install, run: `pip install webrtcvad`
# import webrtcvad

# # another method (GMM + BIC stop)
# from sklearn.mixture import GaussianMixture

# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Set up logging
# logging.basicConfig(level=logging.INFO)

# # Parameters with default values
# MERGE_CONSECUTIVE_SEGMENTS = True
# WINDOW_SIZE = 1.0          # in seconds
# HOP_SIZE = 0.75            # in seconds
# SMOOTHING_WINDOW_SIZE = 5
# WHISPER_MODEL_NAME = "turbo"  # Options: "tiny", "base", "small", "medium", "medium.en", "large", "large-v3", "turbo"

# def load_audio(filepath, target_sr=16000):
#     # Load audio with librosa
#     audio, sr = librosa.load(filepath, sr=target_sr)
#     return audio, sr

# def segment_audio(audio, sr, window_size=WINDOW_SIZE, hop_size=HOP_SIZE):
#     # Split audio into overlapping windows
#     window_length = int(window_size * sr)
#     hop_length = int(hop_size * sr)
#     segments = []
#     timestamps = []
#     for start in range(0, len(audio) - window_length + 1, hop_length):
#         end = start + window_length
#         segments.append(audio[start:end])
#         timestamps.append((start / sr, end / sr))
#     return segments, timestamps

# def get_embeddings(segments, encoder):
#     embeddings = []
#     for seg in segments:
#         emb = encoder.embed_utterance(seg)
#         embeddings.append(emb)
#     embeddings = np.vstack(embeddings)
#     return embeddings

# def smooth_labels(labels, window_size=SMOOTHING_WINDOW_SIZE):
#     smoothed_labels = uniform_filter1d(labels.astype(float), size=window_size, mode='nearest')
#     smoothed_labels = np.round(smoothed_labels).astype(int)
#     return smoothed_labels

# # VAD masking with webrtcvad
# def vad_mask(audio, sr, frame_ms=30, aggressiveness=2):
#     vad = webrtcvad.Vad(aggressiveness)
#     frame = int(sr * frame_ms / 1000)
#     # pad to frame multiple
#     pad = (-len(audio)) % frame
#     if pad: audio = np.pad(audio, (0,pad))
#     pcm16 = (np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes()
#     voiced = []
#     for i in range(0, len(pcm16), frame*2):
#         chunk = pcm16[i:i+frame*2]
#         voiced.append(vad.is_speech(chunk, sr))
#     return np.repeat(np.array(voiced, dtype=bool), frame)[:len(audio)]

# # old estimation method for number of speakers
# def estimate_num_speakers(embeddings, min_speakers=1, max_speakers=10):
#     """
#     Estimate the optimal number of speakers using silhouette score.
#     """
#     best_score = -1
#     best_num_speakers = min_speakers
#     best_labels = np.zeros(len(embeddings))  # Default to one speaker

#     # If only one speaker, just return one label for all
#     if min_speakers == 1 and max_speakers == 1:
#         logging.info(f"Detected one speaker, skipping clustering.")
#         return best_labels

#     for n_speakers in range(min_speakers, max_speakers + 1):
#         # Perform clustering with a different number of speakers
#         refinement_options = RefinementOptions(
#             p_percentile=0.90,
#             gaussian_blur_sigma=1,
#         )
#         clusterer = SpectralClusterer(
#             min_clusters=n_speakers,
#             max_clusters=n_speakers,
#             refinement_options=refinement_options,
#         )
#         labels = clusterer.predict(embeddings)

#         # Skip silhouette score if only 1 cluster
#         if len(np.unique(labels)) == 1:
#             logging.info(f"Only one cluster found with {n_speakers} speakers. Skipping silhouette score.")
#             continue

#         # Calculate the silhouette score for this clustering
#         score = silhouette_score(embeddings, labels)

#         logging.info(f"Number of speakers: {n_speakers}, Silhouette Score: {score}")

#         # Keep track of the best score and corresponding number of speakers
#         if score > best_score:
#             best_score = score
#             best_num_speakers = n_speakers
#             best_labels = labels

#     logging.info(f"Best number of speakers: {best_num_speakers} with a Silhouette Score of {best_score}")
#     return best_labels

# # new method for estimating number of speakers; GMM + BIC stop
# # atm not fully implemented
# def cluster_gmm_bic(embeddings, k_min=1, k_max=12, covariance_type="diag"):
#     X = np.asarray(embeddings, dtype=np.float32)
#     if len(X) < 8:
#         return np.zeros(len(X), dtype=int)  # not enough info → 1 spk
#     bics, gmms = [], []
#     for k in range(k_min, k_max+1):
#         gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=2, random_state=0)
#         gmm.fit(X)
#         bics.append(gmm.bic(X)); gmms.append(gmm)
#     best = int(np.argmin(bics))
#     labels = gmms[best].predict(X)
#     return labels

# def transcribe_audio(filepath, model_name=WHISPER_MODEL_NAME):
#     # Load Whisper model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = whisper.load_model(model_name, device=device)
#     # Transcribe audio
#     result = model.transcribe(filepath)
#     # Get segments
#     segments = result["segments"]
#     return segments

# def assign_speakers_to_transcripts(transcript_segments, diarization_labels, diarization_timestamps):
#     speaker_transcripts = []
#     for segment in transcript_segments:
#         start = segment['start']
#         end = segment['end']
#         text = segment['text']
#         # Find the diarization label(s) for this segment
#         speaker_label = 'Unknown'
#         speaker_counts = {}
#         for idx, (d_start, d_end) in enumerate(diarization_timestamps):
#             # Check for overlap
#             overlap = max(0, min(end, d_end) - max(start, d_start))
#             if overlap > 0:
#                 label = f"Speaker {diarization_labels[idx]+1}"
#                 speaker_counts[label] = speaker_counts.get(label, 0) + overlap
#         if speaker_counts:
#             # Assign the speaker with the maximum overlap
#             speaker_label = max(speaker_counts, key=speaker_counts.get)
#         speaker_transcripts.append({
#             'start': start,
#             'end': end,
#             'speaker': speaker_label,
#             'text': text.strip()
#         })
#     return speaker_transcripts

# def merge_consecutive_speaker_segments(speaker_transcripts):
#     merged_segments = []
#     current_segment = None

#     for segment in speaker_transcripts:
#         if current_segment is None:
#             current_segment = segment.copy()
#         elif segment['speaker'] == current_segment['speaker']:
#             # Merge the text and update the end time
#             current_segment['text'] += ' ' + segment['text']
#             current_segment['end'] = segment['end']
#         else:
#             # Append the current segment and start a new one
#             merged_segments.append(current_segment)
#             current_segment = segment.copy()

#     # Append the last segment
#     if current_segment is not None:
#         merged_segments.append(current_segment)

#     return merged_segments

# def format_output(speaker_transcripts):
#     lines = []
#     for segment in speaker_transcripts:
#         start_time = format_timestamp(segment['start'])
#         speaker = segment['speaker']
#         text = segment['text']
#         lines.append(f"=== {start_time} ({speaker}) ===\n{text}\n")
#     return "\n".join(lines)

# def format_timestamp(seconds):
#     # Format seconds into hh:mm:ss format
#     hours = int(seconds // 3600)
#     minutes = int((seconds % 3600) // 60)
#     secs = int(seconds % 60)
#     if hours > 0:
#         return f"{hours:02d}:{minutes:02d}:{secs:02d}"
#     else:
#         return f"{minutes:02d}:{secs:02d}"

# def main(audio_filepath, output_filepath=None):
#     logging.info("Loading audio file...")
#     audio, sr = load_audio(audio_filepath)
#     logging.info("Segmenting audio...")
#     segments, timestamps = segment_audio(audio, sr)
#     logging.info("Computing embeddings...")
#     encoder = VoiceEncoder()
#     embeddings = get_embeddings(segments, encoder)
#     logging.info("Estimating number of speakers and clustering embeddings...")
#     labels = estimate_num_speakers(embeddings)  # Dynamic speaker estimation
#     logging.info("Smoothing labels...")
#     labels = smooth_labels(labels)
#     logging.info("Transcribing audio...")
#     transcript_segments = transcribe_audio(audio_filepath)
#     logging.info("Assigning speaker labels to transcripts...")
#     speaker_transcripts = assign_speakers_to_transcripts(transcript_segments, labels, timestamps)

#     if MERGE_CONSECUTIVE_SEGMENTS:
#         logging.info("Merging consecutive segments with the same speaker...")
#         speaker_transcripts = merge_consecutive_speaker_segments(speaker_transcripts)

#     output = format_output(speaker_transcripts)
#     logging.info("=== Diarization and Transcription Output ===\n")
#     print(output)

#     if output_filepath:
#         with open(output_filepath, 'w', encoding='utf-8') as f:
#             f.write(output)
#         logging.info(f"\nTranscription has been saved to {output_filepath}")

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python diarization_resemblyzer.py path_to_audio_file [output_textfile.txt]")
#         sys.exit(1)
#     audio_filepath = sys.argv[1]
#     output_filepath = sys.argv[2] if len(sys.argv) > 2 else None
#     main(audio_filepath, output_filepath)
