"""
File watching utilities for dashboard log consumption.

Implements line-buffered log watching optimized for files that are
flushed after each write.
"""

import json
import time
from pathlib import Path
from typing import List, Optional, TextIO


class LineBufferedLogWatcher:
    """
    Watches a line-buffered NDJSON file for new entries.

    Optimized for files that are flushed after each write.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.last_position = 0
        self._file_handle: Optional[TextIO] = None

    def poll(self) -> List[dict]:
        """
        Return new lines since last poll.

        Returns empty list if file doesn't exist or no new data.
        """
        if not self.path.exists():
            return []

        try:
            # Open file if not already open
            if self._file_handle is None:
                self._file_handle = open(self.path, 'r')

            # Seek to last known position
            self._file_handle.seek(self.last_position)

            # Read new lines
            new_lines = []
            for line in self._file_handle:
                line = line.strip()
                if line:
                    try:
                        new_lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # Skip malformed lines

            # Update position
            self.last_position = self._file_handle.tell()

            return new_lines

        except IOError:
            # File was rotated/deleted, reset
            self._file_handle = None
            self.last_position = 0
            return []

    def reset(self) -> None:
        """Reset watcher state (e.g., when switching missions)."""
        if self._file_handle:
            self._file_handle.close()
        self._file_handle = None
        self.last_position = 0

    def read_all(self) -> List[dict]:
        """Read all entries from file (for replay mode)."""
        if not self.path.exists():
            return []

        entries = []
        try:
            with open(self.path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except IOError:
            pass

        return entries


def detect_mission_state(tick_state_path: str = "./logs/tick_state.jsonl") -> str:
    """
    Detect current mission state based on file existence and modification time.

    Returns:
        "NO_MISSION" - No log files exist
        "LIVE" - Mission is actively running (file modified in last 2 seconds)
        "COMPLETED" - Mission has finished (file exists but not recently modified)
    """
    tick_path = Path(tick_state_path)
    if not tick_path.exists():
        return "NO_MISSION"

    # Check if file is still being written (modified in last 2 seconds)
    mtime = tick_path.stat().st_mtime
    if time.time() - mtime < 2.0:
        return "LIVE"
    else:
        return "COMPLETED"


def list_archived_missions(archive_dir: str = "./logs/archive") -> List[str]:
    """
    List available archived missions.

    Returns:
        List of mission directory names, sorted newest first.
    """
    archive_path = Path(archive_dir)
    if not archive_path.exists():
        return []

    missions = []
    for d in archive_path.iterdir():
        if d.is_dir() and d.name.startswith("mission_"):
            missions.append(d.name)

    return sorted(missions, reverse=True)
