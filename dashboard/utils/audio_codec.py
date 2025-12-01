"""
Audio encoding/decoding utilities for dashboard.

Handles base64+gzip compressed audio data from tick state logs.
"""

import base64
import gzip
import sys
from pathlib import Path
from collections import deque
from typing import Optional

import numpy as np

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import CHUNK_SAMPLES, CONTEXT_WINDOW, CHUNK_DURATION
from dsp_pipeline import generate_spectrogram


def decode_audio(audio_b64gz: str) -> Optional[np.ndarray]:
    """
    Decode gzip-compressed base64 audio back to numpy array.

    Args:
        audio_b64gz: Base64-encoded gzip-compressed audio string

    Returns:
        Audio array as float32, or None if input is None/invalid
    """
    if audio_b64gz is None:
        return None

    try:
        compressed = base64.b64decode(audio_b64gz)
        raw_bytes = gzip.decompress(compressed)
        # Was stored as float16
        audio = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)
        return audio
    except Exception:
        return None


def render_spectrogram(audio_b64gz: str) -> Optional[np.ndarray]:
    """
    Generate spectrogram from serialized audio for display.

    Args:
        audio_b64gz: Base64-encoded gzip-compressed audio string

    Returns:
        2D spectrogram array [224, 224] for heatmap display, or None if invalid
    """
    audio = decode_audio(audio_b64gz)
    if audio is None:
        return None

    # Derive chunk count from config (typically 3 chunks)
    num_chunks = int(CONTEXT_WINDOW / CHUNK_DURATION)

    # Split concatenated audio back into chunks using config-derived size
    chunks = [audio[i:i+CHUNK_SAMPLES] for i in range(0, len(audio), CHUNK_SAMPLES)]

    # Ensure we have the right number of chunks
    if len(chunks) < num_chunks:
        return None

    buffer = deque(chunks[:num_chunks], maxlen=num_chunks)

    try:
        spectrogram = generate_spectrogram(buffer)
        # Return 2D array for heatmap (drop batch and channel dims)
        return spectrogram[0, 0, :, :]  # [224, 224]
    except Exception:
        return None
