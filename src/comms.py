"""
Telemetry and communications module for marine sensor node.

Handles NDJSON logging of detection events and system status.
Per spec: Max 340 bytes per event (RockBLOCK SBD limit).
"""

import json
import os
import base64
import gzip
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

from config import LOG_FILE, CLASS_NAMES


class TelemetryLogger:
    """
    NDJSON telemetry logger.

    Per spec Section D:
    - Format: NDJSON (Newline Delimited JSON)
    - File: logs/mission_log.jsonl (Append mode)
    - Max payload: 340 bytes per event
    - Event types: VESSEL, CETACEAN, LOW_BATTERY_SHUTDOWN, MISSION_END
    """

    def __init__(self, log_path: str = LOG_FILE):
        """
        Initialize logger.

        Args:
            log_path: Path to NDJSON log file
        """
        self.log_path = Path(log_path)
        self._file_handle: Optional[Any] = None
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure log directory exists."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def open(self) -> None:
        """Open log file for appending."""
        # Clear any existing log for fresh simulation run
        if self.log_path.exists():
            self.log_path.unlink()
        self._file_handle = open(self.log_path, 'a')

    def close(self) -> None:
        """Close log file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def log_event(self, event: Dict[str, Any]) -> None:
        """
        Log an event to the NDJSON file.

        Args:
            event: Dictionary containing event data

        Event format:
            {"ts": int, "batt_v": float, "event": str, ...}
        """
        if self._file_handle is None:
            self.open()

        line = json.dumps(event, separators=(',', ':'))  # Compact format
        self._file_handle.write(line + '\n')
        self._file_handle.flush()  # Ensure dashboard sees events immediately

    def flush(self) -> None:
        """Flush buffered writes to disk."""
        if self._file_handle:
            self._file_handle.flush()
            os.fsync(self._file_handle.fileno())

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self.close()


def log_detection(
    logger: TelemetryLogger,
    tick: int,
    voltage: float,
    class_id: int,
    confidence: float,
    rms: float
) -> None:
    """
    Log a detection event.

    Args:
        logger: TelemetryLogger instance
        tick: Current tick count
        voltage: Battery voltage
        class_id: Detected class (1=Vessel, 2=Cetacean)
        confidence: Model confidence (0.0-1.0)
        rms: RMS of triggering audio chunk
    """
    # Map class_id to event name
    if class_id == 1:
        event_name = "VESSEL"
    elif class_id == 2:
        event_name = "CETACEAN"
    else:
        # Should not happen for valid detections
        event_name = "UNKNOWN"

    event = {
        "ts": tick,
        "batt_v": round(voltage, 2),
        "event": event_name,
        "conf": round(confidence, 2),
        "rms": round(rms, 2)
    }

    logger.log_event(event)


def log_shutdown(
    logger: TelemetryLogger,
    tick: int,
    voltage: float,
    reason: str
) -> None:
    """
    Log a shutdown event.

    Args:
        logger: TelemetryLogger instance
        tick: Current tick count
        voltage: Battery voltage at shutdown
        reason: "LOW_BATTERY" or "AUDIO_EXHAUSTED"
    """
    if reason == "LOW_BATTERY":
        event_name = "LOW_BATTERY_SHUTDOWN"
    else:
        event_name = "MISSION_END"

    event = {
        "ts": tick,
        "batt_v": round(voltage, 2),
        "event": event_name
    }

    logger.log_event(event)


class TickStateLogger:
    """
    Per-tick state logger for dashboard visualization.

    Writes line-buffered NDJSON with optional compressed audio data.
    Audio is gzip-compressed and base64-encoded to reduce file size.
    """

    def __init__(self, log_path: str = None, include_audio: bool = None):
        from config import TICK_STATE_FILE, DASHBOARD_AUDIO_ENABLED
        self.log_path = Path(log_path if log_path is not None else TICK_STATE_FILE)
        self.include_audio = include_audio if include_audio is not None else DASHBOARD_AUDIO_ENABLED
        self._file_handle = None
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def open(self) -> None:
        """Open with line buffering for immediate visibility."""
        if self.log_path.exists():
            self.log_path.unlink()
        self._file_handle = open(self.log_path, 'w', buffering=1)  # Line-buffered

    def close(self) -> None:
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def _encode_audio(self, chunks: List[np.ndarray]) -> Optional[str]:
        """Compress and encode audio chunks for JSON serialization."""
        if not chunks or not self.include_audio:
            return None
        # Stack chunks into single array, convert to float16 to halve size
        stacked = np.concatenate(chunks).astype(np.float16)
        compressed = gzip.compress(stacked.tobytes(), compresslevel=1)
        return base64.b64encode(compressed).decode('ascii')

    def log_tick(
        self,
        tick: int,
        state_at_entry: str,       # State when tick began (for power/timeline consistency)
        state_at_exit: str,        # State when tick ended (for next-tick prediction)
        voltage: float,
        capacity_pct: float,
        current_ma: int,           # Power consumed THIS tick (matches state_at_entry)
        buffer_fill: int,          # Current buffer size at END of tick (from live buffer)
        buffer_chunks_for_audio: List[np.ndarray],  # Snapshot for audio (may differ from buffer_fill on detection ticks)
        rms: Optional[float],
        tx_timer: int,
        sleep_timer: int,
        inference_run: bool,
        class_id: Optional[int] = None,
        confidence: Optional[float] = None,
        probabilities: Optional[List[float]] = None
    ) -> None:
        """
        Log tick state with optional audio data.

        Timing semantics:
        - state_at_entry: State at START of tick (used for power calculation, timeline)
        - state_at_exit: State at END of tick (after all transitions)
        - current_ma: Power consumed during this tick (corresponds to state_at_entry)
        - voltage/capacity_pct: Values AFTER power consumption
        - buffer_fill: Live buffer size at END of tick (0 after detection clears buffer)
        - buffer_chunks_for_audio: Pre-clear snapshot used ONLY for audio encoding

        This ensures the power panel and timeline are consistent: if state_at_entry
        is LISTENING and current_ma is 120, the dashboard knows this tick consumed
        120mA in LISTENING state, even if state_at_exit is TRANSMIT.

        On detection ticks: buffer_fill=0 (cleared), but buffer_chunks_for_audio
        contains the 3 chunks that triggered detection (for waveform/spectrogram).
        """
        if self._file_handle is None:
            self.open()

        event = {
            "tick": tick,
            "state_at_entry": state_at_entry,
            "state_at_exit": state_at_exit,
            "voltage": round(voltage, 3),
            "capacity_pct": round(capacity_pct, 2),
            "current_ma": current_ma,
            "buffer_fill": buffer_fill,  # From live buffer, not snapshot
            "rms": round(rms, 4) if rms is not None else None,
            "tx_timer": tx_timer,
            "sleep_timer": sleep_timer,
            "inference_run": inference_run,
            "class_id": class_id,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "probabilities": [round(p, 4) for p in probabilities] if probabilities else None,
            "audio_b64gz": self._encode_audio(buffer_chunks_for_audio) if inference_run else None
        }

        line = json.dumps(event, separators=(',', ':'))
        self._file_handle.write(line + '\n')
        # Line buffering handles flush automatically

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def archive_mission_logs() -> Optional[str]:
    """
    Archive existing mission logs before starting new run.

    Returns:
        Archive directory path if logs were archived, None if no logs existed.
    """
    from config import LOG_FILE, TICK_STATE_FILE, LOG_ARCHIVE_DIR

    log_path = Path(LOG_FILE)
    tick_path = Path(TICK_STATE_FILE)
    archive_dir = Path(LOG_ARCHIVE_DIR)

    # Check if any logs exist to archive
    if not log_path.exists() and not tick_path.exists():
        return None

    # Create timestamped archive subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mission_archive = archive_dir / f"mission_{timestamp}"
    mission_archive.mkdir(parents=True, exist_ok=True)

    # Move existing logs to archive
    if log_path.exists():
        log_path.rename(mission_archive / log_path.name)
    if tick_path.exists():
        tick_path.rename(mission_archive / tick_path.name)

    return str(mission_archive)
