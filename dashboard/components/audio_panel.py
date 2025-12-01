"""
Audio panel component for dashboard.

Displays waveform visualization, RMS meter, and buffer indicator.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from audio_codec import decode_audio

from config import CHUNK_DURATION, CHUNK_SAMPLES, CONTEXT_WINDOW, WAKE_THRESHOLD


def render_waveform(audio: Optional[np.ndarray]) -> None:
    """
    Render waveform visualization.

    Args:
        audio: Audio array or None if no data
    """
    if audio is None or len(audio) == 0:
        st.info("Waiting for inference trigger...")
        return

    # Downsample for display (show every Nth sample)
    display_samples = 2000
    if len(audio) > display_samples:
        step = len(audio) // display_samples
        audio_display = audio[::step]
    else:
        audio_display = audio

    # Time axis
    total_duration = len(audio) / (CHUNK_SAMPLES / CHUNK_DURATION)
    time_axis = np.linspace(0, total_duration, len(audio_display))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=audio_display,
            mode="lines",
            line=dict(color="#06b6d4", width=1),
            fill="tozeroy",
            fillcolor="rgba(6, 182, 212, 0.2)",
        )
    )

    fig.update_layout(
        height=150,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="Time (s)"),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            range=[-1.1, 1.1],
            title="Amplitude",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, width="stretch")


def render_rms_meter(rms: Optional[float]) -> None:
    """
    Render RMS meter with threshold indicator.

    Args:
        rms: Current RMS value or None
    """
    st.markdown("**RMS Level**")

    if rms is None:
        st.progress(0.0)
        st.caption("N/A (blind period)")
        return

    # Normalize for display (RMS typically 0-0.5 range)
    normalized_rms = min(rms / 0.5, 1.0)
    st.progress(normalized_rms)

    # Show value and threshold comparison
    above_threshold = rms > WAKE_THRESHOLD
    color = "#10b981" if above_threshold else "#64748b"
    status = "Above threshold" if above_threshold else "Below threshold"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**RMS:** {rms:.4f}")
    with col2:
        st.markdown(
            f"<span style='color: {color};'>{status}</span>", unsafe_allow_html=True
        )

    st.caption(f"Threshold: {WAKE_THRESHOLD}")


def render_buffer_indicator(buffer_fill: int) -> None:
    """
    Render buffer fill indicator (3 squares).

    Args:
        buffer_fill: Number of chunks in buffer (0-3)
    """
    st.markdown("**Buffer**")

    # Calculate number of chunks from config
    num_chunks = int(CONTEXT_WINDOW / CHUNK_DURATION)

    # Build visual indicator
    squares = []
    for i in range(num_chunks):
        if i < buffer_fill:
            squares.append(
                "<span style='display: inline-block; width: 24px; height: 24px; "
                "background-color: #10b981; border-radius: 3px; margin-right: 4px;'></span>"
            )
        else:
            squares.append(
                "<span style='display: inline-block; width: 24px; height: 24px; "
                "background-color: #1e293b; border: 1px solid #334155; "
                "border-radius: 3px; margin-right: 4px;'></span>"
            )

    st.markdown("".join(squares), unsafe_allow_html=True)
    st.caption(f"{buffer_fill}/{num_chunks} chunks")


def render_audio_panel(
    current_state: Optional[Dict[str, Any]], last_inference_audio: Optional[str]
) -> None:
    """
    Render the audio panel.

    Args:
        current_state: Most recent tick state dict
        last_inference_audio: Base64-encoded audio from last inference
    """
    st.markdown("### Audio")

    # Waveform - show last inference audio or placeholder
    st.markdown("**Waveform** (3-second window)")
    audio = decode_audio(last_inference_audio) if last_inference_audio else None
    render_waveform(audio)

    # RMS and Buffer in columns
    col1, col2 = st.columns(2)

    with col1:
        rms = current_state.get("rms") if current_state else None
        render_rms_meter(rms)

    with col2:
        buffer_fill = current_state.get("buffer_fill", 0) if current_state else 0
        render_buffer_indicator(buffer_fill)
