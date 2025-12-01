"""
Hardware simulation module for marine sensor node.

Simulates:
- Battery model with linear voltage curve (3.0V-4.2V)
- Audio environment with file streaming from sim_input/
- Power consumption based on state (LISTENING/TRANSMIT/SLEEP)
"""

import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Optional, List, Iterator

from config import (
    SAMPLE_RATE, CHUNK_DURATION, BATTERY_CAPACITY_MAH,
    POWER_CONSUMPTION, SIM_INPUT_DIR
)


# Derived constants
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)  # 16000 samples per chunk


class Battery:
    """
    Simulates 1S LiPo battery with linear discharge model.

    Per spec Section 3:
    - Voltage range: 3.0V (empty) to 4.2V (full)
    - Capacity: 10000 mAh default
    - Linear voltage model: V = 3.0 + 1.2 * (capacity_ratio)
    - Low-voltage cutoff: V <= 3.2V triggers shutdown
    """

    def __init__(self, capacity_mah: float = BATTERY_CAPACITY_MAH):
        """
        Initialize battery at full charge.

        Args:
            capacity_mah: Total battery capacity in mAh
        """
        self.max_capacity_mah = capacity_mah
        self.current_capacity_mah = capacity_mah
        self._voltage = 4.2  # Start at full charge

    @property
    def voltage(self) -> float:
        """Current battery voltage."""
        return self._voltage

    @property
    def capacity_percent(self) -> float:
        """Current capacity as percentage of max."""
        return 100.0 * self.current_capacity_mah / self.max_capacity_mah

    @property
    def capacity_ratio(self) -> float:
        """Current capacity as ratio (0.0 to 1.0)."""
        return self.current_capacity_mah / self.max_capacity_mah

    def consume_power(self, state: str) -> None:
        """
        Consume power for one tick based on current state.

        Per spec: Called at START of each tick.

        Args:
            state: Current state name ("LISTENING", "TRANSMIT", or "SLEEP")
        """
        current_ma = POWER_CONSUMPTION.get(state, POWER_CONSUMPTION["LISTENING"])
        # delta_mah = current_ma * (seconds / 3600)
        delta_mah = current_ma * (CHUNK_DURATION / 3600.0)
        self.current_capacity_mah = max(0, self.current_capacity_mah - delta_mah)

    def update_voltage(self) -> None:
        """
        Update voltage based on current capacity.

        Per spec: Called AFTER consuming power each tick.
        Linear model: V = 3.0 + 1.2 * ratio
        """
        ratio = self.capacity_ratio
        self._voltage = 3.0 + 1.2 * ratio

    def is_low(self) -> bool:
        """Check if voltage is at or below cutoff threshold."""
        return self._voltage <= 3.2

    def mah_consumed(self) -> float:
        """Total mAh consumed since start."""
        return self.max_capacity_mah - self.current_capacity_mah


class AudioEnvironment:
    """
    Simulates audio input environment by streaming from files.

    Per spec Section A:
    - Scans sim_input/ for .wav and .flac files
    - Files sorted alphabetically, processed in order
    - Resamples to 16kHz if needed
    - Converts stereo to mono
    - Zero-pads partial final chunks
    - Cross-file continuity (files treated as one stream)
    - Returns None when all files exhausted
    """

    SUPPORTED_EXTENSIONS = {'.wav', '.flac'}

    def __init__(self, input_dir: str = SIM_INPUT_DIR):
        """
        Initialize audio environment.

        Args:
            input_dir: Directory containing audio files

        Raises:
            FileNotFoundError: If directory doesn't exist or contains no audio
        """
        self.input_dir = Path(input_dir)
        self._file_queue: List[Path] = []
        self._current_file: Optional[Path] = None
        self._current_data: Optional[np.ndarray] = None
        self._file_position: int = 0
        self._exhausted: bool = False

        self._scan_input_directory()

    def _scan_input_directory(self) -> None:
        """Scan input directory for audio files."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        # Find all supported audio files, sorted alphabetically
        audio_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            audio_files.extend(self.input_dir.glob(f"*{ext}"))

        self._file_queue = sorted(audio_files)

        if not self._file_queue:
            raise FileNotFoundError(
                f"No audio files found in {self.input_dir}. "
                f"Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )

        print(f"Found {len(self._file_queue)} audio file(s):")
        for f in self._file_queue:
            print(f"  - {f.name}")

    def _load_next_file(self) -> bool:
        """
        Load the next file from the queue.

        Returns:
            True if a file was loaded, False if queue is empty
        """
        if not self._file_queue:
            return False

        self._current_file = self._file_queue.pop(0)
        self._file_position = 0

        try:
            # Load audio file
            data, sr = sf.read(self._current_file, dtype='float32')

            # Handle stereo -> mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Resample if not 16kHz
            if sr != SAMPLE_RATE:
                print(f"  Resampling {self._current_file.name} from {sr}Hz to {SAMPLE_RATE}Hz")
                data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)

            self._current_data = data.astype(np.float32)
            print(f"Loaded: {self._current_file.name} ({len(self._current_data)/SAMPLE_RATE:.1f}s)")
            return True

        except Exception as e:
            print(f"Warning: Failed to load {self._current_file}: {e}")
            # Try next file
            return self._load_next_file()

    def read_chunk(self) -> Optional[np.ndarray]:
        """
        Read the next 1-second chunk of audio.

        Per spec:
        - Returns 16000 samples (1 second at 16kHz)
        - Zero-pads partial final chunks
        - Continues across file boundaries
        - Returns None when all audio exhausted

        Returns:
            np.ndarray of shape [16000], float32, or None if exhausted
        """
        if self._exhausted:
            return None

        # Load first/next file if needed
        if self._current_data is None:
            if not self._load_next_file():
                self._exhausted = True
                return None

        # Calculate remaining samples in current file
        remaining = len(self._current_data) - self._file_position

        if remaining >= CHUNK_SAMPLES:
            # Full chunk available
            chunk = self._current_data[self._file_position:self._file_position + CHUNK_SAMPLES]
            self._file_position += CHUNK_SAMPLES
            return chunk

        elif remaining > 0:
            # Partial chunk from the end of the current file.
            # Spec: zero-pad the remainder to a full tick and treat as its own chunk.
            chunk = self._current_data[self._file_position:].copy()
            self._file_position = len(self._current_data)

            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

            # Do NOT pull data from the next file here; the next tick will read it.
            # Mark current file exhausted; next call will load the next file if present.
            return chunk.astype(np.float32)

        else:
            # Current file exhausted, try next
            if self._load_next_file():
                return self.read_chunk()
            else:
                self._exhausted = True
                return None

    def is_exhausted(self) -> bool:
        """Check if all audio has been read."""
        return self._exhausted

    def reset(self) -> None:
        """Reset environment to beginning (re-scan files)."""
        self._exhausted = False
        self._current_data = None
        self._current_file = None
        self._file_position = 0
        self._scan_input_directory()
