"""
Telemetry and communications module for marine sensor node.

Handles NDJSON logging of detection events and system status.
Per spec: Max 340 bytes per event (RockBLOCK SBD limit).
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

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
