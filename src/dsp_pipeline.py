"""
DSP Pipeline for marine acoustic anomaly detection.

Handles:
- RMS calculation on raw audio (power gating)
- Audio normalization with silence protection
- Mel-spectrogram generation with exact parameters per spec
- Buffer management for 3-second context window
"""

import numpy as np
import librosa
from collections import deque
from typing import Optional

from config import (
    SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, TIME_STEPS,
    CHUNK_DURATION, CONTEXT_WINDOW
)


# Derived constants
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)  # 16000 samples per chunk
CONTEXT_CHUNKS = int(CONTEXT_WINDOW / CHUNK_DURATION)  # 3 chunks


def compute_rms(raw_chunk: np.ndarray) -> float:
    """
    Compute RMS on raw (pre-normalized) audio.

    Per spec Section C.1: RMS must be computed on raw audio to reflect
    true acoustic amplitude. Used for power gating before inference.

    Args:
        raw_chunk: 1-second audio array (16000 samples), float32, [-1.0, 1.0]

    Returns:
        RMS value (0.0 to ~1.0 range)
    """
    return float(np.sqrt(np.mean(raw_chunk ** 2)))


def normalize(raw_chunk: np.ndarray) -> np.ndarray:
    """
    Peak-normalize audio with silence protection.

    Per spec Section 4.1: Normalize with silence protection to avoid
    division by zero for silent audio.

    Args:
        raw_chunk: Raw audio array, float32

    Returns:
        Normalized audio array, float32, peak at 1.0 (or zeros if silent)
    """
    max_val = np.max(np.abs(raw_chunk))
    if max_val > 1e-6:  # Avoid division by zero for silent audio
        return raw_chunk / max_val
    else:
        return np.zeros_like(raw_chunk)  # Treat as silence


def generate_spectrogram(buffer: deque) -> np.ndarray:
    """
    Generate mel-spectrogram from 3-second buffer.

    Per spec Section 4.2:
    - Concatenate 3 x 1-second normalized chunks
    - Apply mel-spectrogram with exact parameters
    - Convert to log scale, normalize to [0,1]
    - Pad to 224x224, stack to 3 channels

    Args:
        buffer: deque containing 3 normalized 1-second audio chunks

    Returns:
        np.ndarray of shape [1, 3, 224, 224], float32, values in [0, 1]
    """
    if len(buffer) != CONTEXT_CHUNKS:
        raise ValueError(f"Buffer must contain {CONTEXT_CHUNKS} chunks, got {len(buffer)}")

    # Concatenate buffer chunks (3 seconds = 48000 samples)
    audio_3s = np.concatenate(list(buffer))

    # Generate mel-spectrogram with exact parameters from spec
    S = librosa.feature.melspectrogram(
        y=audio_3s,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        window='hann',
        center=False,
        power=2.0,
        fmin=20.0,
        fmax=8000.0
    )

    # Convert to log scale (dB)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalize to [0, 1]
    S_min = S_db.min()
    S_max = S_db.max()
    S_norm = (S_db - S_min) / (S_max - S_min + 1e-6)

    # Pad to TIME_STEPS (224) frames
    # With hop_length=214 and 48000 samples: floor((48000-1024)/214)+1 = 220 frames
    current_frames = S_norm.shape[1]
    if current_frames < TIME_STEPS:
        pad_width = TIME_STEPS - current_frames
        S_norm = np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant')
    elif current_frames > TIME_STEPS:
        S_norm = S_norm[:, :TIME_STEPS]

    # Final shape should be [224, 224]
    assert S_norm.shape == (N_MELS, TIME_STEPS), f"Unexpected shape: {S_norm.shape}"

    # Stack 3x for RGB channels and add batch dimension
    # Shape: [1, 3, 224, 224] - NCHW format
    S_stacked = np.stack([S_norm, S_norm, S_norm], axis=0)
    S_batch = np.expand_dims(S_stacked, axis=0)

    return S_batch.astype(np.float32)


class AudioBuffer:
    """
    Manages the 3-second sliding window buffer for inference.

    Per spec Section B:
    - Uses collections.deque(maxlen=3) storing 1-second numpy arrays
    - Buffer is cleared on entering TRANSMIT and on exiting SLEEP
    - System must wait 3 ticks to refill before inference can trigger
    """

    def __init__(self):
        self.buffer: deque = deque(maxlen=CONTEXT_CHUNKS)

    def append(self, normalized_chunk: np.ndarray) -> None:
        """Add a normalized 1-second chunk to the buffer."""
        self.buffer.append(normalized_chunk)

    def clear(self) -> None:
        """Clear the buffer (called on TRANSMIT entry and SLEEP exit)."""
        self.buffer.clear()

    def is_full(self) -> bool:
        """Check if buffer has 3 chunks ready for inference."""
        return len(self.buffer) == CONTEXT_CHUNKS

    def get_spectrogram(self) -> np.ndarray:
        """Generate spectrogram from current buffer contents."""
        return generate_spectrogram(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)
