"""
State machine panel component for dashboard.

Displays state diagram with current state highlight and timeline visualization.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))


# State colors - professional muted tones
STATE_COLORS = {
    "LISTENING": "#10b981",  # Emerald
    "INFERENCE": "#3b82f6",  # Blue
    "TRANSMIT": "#f59e0b",  # Amber
    "SLEEP": "#64748b",  # Slate
    "SHUTDOWN": "#ef4444",  # Red
}


def build_timeline_segments(tick_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build timeline segments from tick history.

    Groups consecutive ticks with the same state_at_entry into segments.

    Args:
        tick_history: List of tick state dicts

    Returns:
        List of segment dicts with start, end, and state_at_entry
    """
    if not tick_history:
        return []

    segments = []
    current_segment = None

    for tick_state in tick_history:
        state = tick_state.get("state_at_entry", "UNKNOWN")
        tick = tick_state.get("tick", 0)

        if current_segment is None:
            current_segment = {"start": tick, "end": tick, "state_at_entry": state}
        elif state == current_segment["state_at_entry"]:
            current_segment["end"] = tick
        else:
            segments.append(current_segment)
            current_segment = {"start": tick, "end": tick, "state_at_entry": state}

    if current_segment:
        segments.append(current_segment)

    return segments


def calculate_coverage_stats(tick_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate coverage statistics from tick history.

    Per spec Section 6.2:
    - blind_ticks: Count where state_at_entry is TRANSMIT or SLEEP
    - active_ticks: Total - blind_ticks

    Args:
        tick_history: List of tick state dicts

    Returns:
        Dict with active_ticks, blind_ticks, active_pct, blind_pct
    """
    if not tick_history:
        return {"active_ticks": 0, "blind_ticks": 0, "active_pct": 0, "blind_pct": 0}

    total = len(tick_history)
    blind_ticks = sum(
        1 for t in tick_history if t.get("state_at_entry") in ("TRANSMIT", "SLEEP")
    )
    active_ticks = total - blind_ticks

    return {
        "active_ticks": active_ticks,
        "blind_ticks": blind_ticks,
        "active_pct": 100.0 * active_ticks / total if total > 0 else 0,
        "blind_pct": 100.0 * blind_ticks / total if total > 0 else 0,
    }


def render_state_diagram(current_state: str) -> None:
    """Render the state machine diagram with current state highlighted."""
    # State positions for layout
    positions = {
        "LISTENING": (0, 1),
        "INFERENCE": (1, 1),
        "TRANSMIT": (2, 1),
        "SLEEP": (2, 0),
    }

    fig = go.Figure()

    # Draw transitions (arrows as lines)
    transitions = [
        ("LISTENING", "INFERENCE"),
        ("INFERENCE", "LISTENING"),
        ("INFERENCE", "TRANSMIT"),
        ("TRANSMIT", "SLEEP"),
        ("SLEEP", "LISTENING"),
    ]

    for start, end in transitions:
        x0, y0 = positions[start]
        x1, y1 = positions[end]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.3)", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Draw state nodes
    for state, (x, y) in positions.items():
        is_current = state == current_state
        color = STATE_COLORS.get(state, "#6b7280")

        # Outer glow for current state
        if is_current:
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    marker=dict(size=60, color=color, opacity=0.3),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Main node
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(
                    size=40 if is_current else 35,
                    color=color,
                    line=dict(width=2, color="white") if is_current else dict(width=0),
                ),
                text=state[:4],
                textposition="middle center",
                textfont=dict(color="white", size=10),
                hoverinfo="text",
                hovertext=state,
                showlegend=False,
            )
        )

    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 2.5]
        ),
        yaxis=dict(
            showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 1.5]
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, width="stretch")


def render_timeline(
    timeline_segments: List[Dict[str, Any]], max_display_ticks: int = 1000
) -> None:
    """
    Render the state timeline as a horizontal stacked bar.

    Args:
        timeline_segments: List of segment dicts from build_timeline_segments
        max_display_ticks: Maximum ticks to show in timeline
    """
    if not timeline_segments:
        st.info("No timeline data yet")
        return

    fig = go.Figure()

    # Calculate total span
    total_start = timeline_segments[0]["start"]
    total_end = timeline_segments[-1]["end"]

    # If too many ticks, show only recent ones
    if total_end - total_start > max_display_ticks:
        display_start = total_end - max_display_ticks
        timeline_segments = [s for s in timeline_segments if s["end"] >= display_start]
        if timeline_segments:
            # Clip first segment
            timeline_segments[0] = {
                **timeline_segments[0],
                "start": max(timeline_segments[0]["start"], display_start),
            }
    else:
        display_start = total_start

    # Draw segments
    for segment in timeline_segments:
        state = segment["state_at_entry"]
        start = segment["start"]
        end = segment["end"]
        width = end - start + 1
        color = STATE_COLORS.get(state, "#6b7280")

        # Add red overlay for blind spots
        if state in ("TRANSMIT", "SLEEP"):
            opacity = 0.8
        else:
            opacity = 1.0

        fig.add_trace(
            go.Bar(
                x=[width],
                y=["Timeline"],
                orientation="h",
                base=start - display_start,
                marker=dict(color=color, opacity=opacity),
                hovertemplate=f"{state}<br>Ticks {start}-{end}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        barmode="stack",
        xaxis=dict(showgrid=False, showticklabels=True, title="Ticks (relative)"),
        yaxis=dict(showgrid=False, showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, width="stretch")


def render_state_machine_panel(
    current_state: Optional[Dict[str, Any]], tick_history: List[Dict[str, Any]]
) -> None:
    """
    Render the full state machine panel.

    Args:
        current_state: Most recent tick state dict
        tick_history: List of all tick states
    """
    st.markdown("### State Machine")

    # Get current state for diagram highlight
    state_name = "LISTENING"
    if current_state:
        state_name = current_state.get("state_at_entry", "LISTENING")

    # State diagram
    render_state_diagram(state_name)

    # Timeline
    st.markdown("**Timeline**")
    segments = build_timeline_segments(tick_history)
    render_timeline(segments)

    # Coverage stats
    stats = calculate_coverage_stats(tick_history)
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Active Time",
            f"{stats['active_ticks']:,} ticks",
            f"{stats['active_pct']:.1f}%",
        )

    with col2:
        st.metric(
            "Blind Time",
            f"{stats['blind_ticks']:,} ticks",
            f"{stats['blind_pct']:.1f}%",
        )
