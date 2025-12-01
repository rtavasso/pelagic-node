"""
Event log panel component for dashboard.

Displays detection events in a sortable, filterable table.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from formatters import format_elapsed

# Event badge colors - professional muted tones
EVENT_COLORS = {
    "VESSEL": "#f59e0b",
    "CETACEAN": "#3b82f6",
    "LOW_BATTERY_SHUTDOWN": "#ef4444",
    "MISSION_END": "#64748b",
}


def render_event_badge(event_type: str) -> str:
    """Generate HTML for event type badge."""
    color = EVENT_COLORS.get(event_type, "#64748b")
    return f"""<span style="
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.03em;
        background-color: {color};
        color: white;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">{event_type}</span>"""


def events_to_dataframe(event_history: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert event history to pandas DataFrame.

    Args:
        event_history: List of event dicts from mission_log

    Returns:
        DataFrame with formatted columns
    """
    if not event_history:
        return pd.DataFrame(
            columns=["Tick", "Time", "Event", "Confidence", "RMS", "Voltage"]
        )

    rows = []
    for event in event_history:
        tick = event.get("ts", 0)
        rows.append(
            {
                "Tick": tick,
                "Time": format_elapsed(tick),
                "Event": event.get("event", "UNKNOWN"),
                "Confidence": f"{event.get('conf', 0):.0%}"
                if event.get("conf")
                else "-",
                "RMS": f"{event.get('rms', 0):.3f}" if event.get("rms") else "-",
                "Voltage": f"{event.get('batt_v', 0):.2f}V",
            }
        )

    return pd.DataFrame(rows)


def render_event_log(
    event_history: List[Dict[str, Any]], filter_type: str = "All"
) -> None:
    """
    Render the event log panel.

    Args:
        event_history: List of event dicts from mission_log
        filter_type: Filter to apply ("All", "Vessels", "Cetaceans", "Shutdown")
    """
    st.markdown("### Event Log")

    # Filter controls
    col1, col2 = st.columns([3, 1])

    with col1:
        filter_options = ["All", "Vessels", "Cetaceans", "Shutdown"]
        selected_filter = st.selectbox(
            "Filter",
            filter_options,
            index=filter_options.index(filter_type)
            if filter_type in filter_options
            else 0,
            key="event_log_filter",
        )

    with col2:
        # Export button
        if event_history:
            df = events_to_dataframe(event_history)
            csv = df.to_csv(index=False)
            st.download_button(
                "Export CSV", csv, file_name="mission_events.csv", mime="text/csv"
            )

    # Apply filter
    filtered_events = event_history
    if selected_filter == "Vessels":
        filtered_events = [e for e in event_history if e.get("event") == "VESSEL"]
    elif selected_filter == "Cetaceans":
        filtered_events = [e for e in event_history if e.get("event") == "CETACEAN"]
    elif selected_filter == "Shutdown":
        filtered_events = [
            e
            for e in event_history
            if e.get("event") in ("LOW_BATTERY_SHUTDOWN", "MISSION_END")
        ]

    if not filtered_events:
        st.info("No events to display")
        return

    # Convert to DataFrame and display
    df = events_to_dataframe(filtered_events)

    # Style the dataframe
    st.dataframe(df, width="stretch", height=300, hide_index=True)

    # Event count summary
    st.caption(f"Showing {len(filtered_events)} of {len(event_history)} events")
