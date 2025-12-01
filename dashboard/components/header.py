"""
Header bar component for dashboard.

Displays current state, tick count, elapsed time, and mission status.
"""

import streamlit as st
from typing import Optional, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from formatters import format_elapsed


# State colors - professional muted tones
STATE_COLORS = {
    "LISTENING": "#10b981",   # Emerald
    "INFERENCE": "#3b82f6",   # Blue
    "TRANSMIT": "#f59e0b",    # Amber
    "SLEEP": "#64748b",       # Slate
    "SHUTDOWN": "#ef4444",    # Red
}

# Mission status colors
MISSION_STATUS_COLORS = {
    "LIVE": "#10b981",        # Emerald
    "COMPLETED": "#64748b",   # Slate
    "NO_MISSION": "#475569",  # Dark slate
}


def render_state_badge(state: str) -> str:
    """Render a colored state badge as HTML."""
    color = STATE_COLORS.get(state, "#64748b")
    return f"""
    <span style="
        display: inline-block;
        padding: 3px 10px;
        border-radius: 3px;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.05em;
        background-color: {color};
        color: white;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">{state}</span>
    """


def render_mission_status_badge(status: str) -> str:
    """Render a mission status badge as HTML."""
    color = MISSION_STATUS_COLORS.get(status, "#64748b")
    display_text = {
        "LIVE": "LIVE",
        "COMPLETED": "COMPLETED",
        "NO_MISSION": "NO MISSION"
    }.get(status, status)

    return f"""
    <span style="
        display: inline-block;
        padding: 3px 10px;
        border-radius: 3px;
        font-weight: 500;
        font-size: 11px;
        letter-spacing: 0.05em;
        background-color: {color}15;
        color: {color};
        border: 1px solid {color}40;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">{display_text}</span>
    """


def render_header(
    current_state: Optional[Dict[str, Any]],
    mission_status: str
) -> None:
    """
    Render the header bar.

    Args:
        current_state: Most recent tick state dict, or None if no data
        mission_status: "LIVE", "COMPLETED", or "NO_MISSION"
    """
    # Use state_at_entry for display (per spec Section 5.2.1)
    if current_state:
        state = current_state.get("state_at_entry", "UNKNOWN")
        tick = current_state.get("tick", 0)
        elapsed = format_elapsed(tick)
    else:
        state = "UNKNOWN"
        tick = 0
        elapsed = "00:00:00"

    # Create header columns
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.markdown("### Pelagic Node Monitor")

    with col2:
        st.markdown("**State**")
        st.markdown(render_state_badge(state), unsafe_allow_html=True)

    with col3:
        st.markdown("**Tick / Time**")
        st.markdown(f"**{tick:,}** | {elapsed}")

    with col4:
        st.markdown("**Mission**")
        st.markdown(render_mission_status_badge(mission_status), unsafe_allow_html=True)

    st.divider()
