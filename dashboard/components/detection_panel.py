"""
Detection panel component for dashboard.

Displays detection counts, confidence history, and last detection details.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from formatters import format_elapsed

# Detection colors - professional muted tones
DETECTION_COLORS = {
    "VESSEL": "#f59e0b",  # Amber
    "CETACEAN": "#3b82f6",  # Blue
}


def render_detection_panel(
    event_history: List[Dict[str, Any]], vessel_count: int, cetacean_count: int
) -> None:
    """
    Render the detection panel.

    Args:
        event_history: List of detection events from mission_log
        vessel_count: Total vessel detections
        cetacean_count: Total cetacean detections
    """
    st.markdown("### Detections")

    total_count = vessel_count + cetacean_count

    # Detection counts in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Vessels**")
        st.markdown(
            f"<span style='color: {DETECTION_COLORS['VESSEL']}; font-size: 32px; font-weight: bold;'>"
            f"{vessel_count}</span>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("**Cetaceans**")
        st.markdown(
            f"<span style='color: {DETECTION_COLORS['CETACEAN']}; font-size: 32px; font-weight: bold;'>"
            f"{cetacean_count}</span>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown("**Total**")
        st.markdown(
            f"<span style='font-size: 32px; font-weight: bold;'>{total_count}</span>",
            unsafe_allow_html=True,
        )

    # Last detection card
    detection_events = [
        e for e in event_history if e.get("event") in ("VESSEL", "CETACEAN")
    ]
    if detection_events:
        last = detection_events[-1]
        event_type = last.get("event", "UNKNOWN")
        tick = last.get("ts", 0)
        conf = last.get("conf", 0)
        color = DETECTION_COLORS.get(event_type, "#6b7280")

        st.markdown("**Last Detection**")
        st.markdown(
            f"""<div style="
                background-color: {color}22;
                border: 1px solid {color};
                border-radius: 8px;
                padding: 12px;
                margin: 8px 0;
            ">
                <span style="color: {color}; font-weight: bold;">{event_type}</span>
                <br>
                <span style="color: #94a3b8;">Tick {tick} | {format_elapsed(tick)} | Conf: {conf:.0%}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # Confidence history scatter plot
    if detection_events:
        st.markdown("**Confidence History**")

        vessel_events = [e for e in detection_events if e.get("event") == "VESSEL"]
        cetacean_events = [e for e in detection_events if e.get("event") == "CETACEAN"]

        fig = go.Figure()

        if vessel_events:
            fig.add_trace(
                go.Scatter(
                    x=[e.get("ts", 0) for e in vessel_events],
                    y=[e.get("conf", 0) for e in vessel_events],
                    mode="markers",
                    name="Vessel",
                    marker=dict(color=DETECTION_COLORS["VESSEL"], size=10),
                )
            )

        if cetacean_events:
            fig.add_trace(
                go.Scatter(
                    x=[e.get("ts", 0) for e in cetacean_events],
                    y=[e.get("conf", 0) for e in cetacean_events],
                    mode="markers",
                    name="Cetacean",
                    marker=dict(color=DETECTION_COLORS["CETACEAN"], size=10),
                )
            )

        # Add confidence threshold line
        fig.add_hline(
            y=0.85,
            line_dash="dash",
            line_color="rgba(255,255,255,0.5)",
            annotation_text="Threshold",
        )

        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="Tick"),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)",
                title="Confidence",
                range=[0.8, 1.0],
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig, width="stretch")
