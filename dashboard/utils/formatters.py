"""
Formatting utilities for dashboard display.

All time calculations use CHUNK_DURATION from config to ensure
consistency with simulation timing.
"""

import sys
from pathlib import Path

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import CHUNK_DURATION


def format_elapsed(tick: int) -> str:
    """
    Format tick count as HH:MM:SS using CHUNK_DURATION.

    Args:
        tick: Current tick count

    Returns:
        Formatted time string "HH:MM:SS"
    """
    total_seconds = tick * CHUNK_DURATION
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_voltage(voltage: float) -> str:
    """Format voltage with unit."""
    return f"{voltage:.2f}V"


def format_percentage(value: float) -> str:
    """Format percentage value."""
    return f"{value:.1f}%"


def format_current(current_ma: int) -> str:
    """Format current draw with unit."""
    return f"{current_ma} mA"


def format_confidence(confidence: float) -> str:
    """Format confidence as percentage."""
    return f"{confidence * 100:.1f}%"
