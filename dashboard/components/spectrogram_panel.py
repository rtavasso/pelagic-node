"""
Spectrogram panel component for dashboard.

Displays spectrogram heatmap, model probability bars, and inference results.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from audio_codec import render_spectrogram

# Class names in spec order
CLASS_NAMES = ["Background", "Vessel", "Cetacean"]

# Class colors - professional muted tones
CLASS_COLORS = {
    "Background": "#64748b",  # Slate
    "Vessel": "#f59e0b",  # Amber
    "Cetacean": "#3b82f6",  # Blue
}


def render_spectrogram_heatmap(spectrogram: Optional[np.ndarray]) -> None:
    """
    Render spectrogram as heatmap.

    Args:
        spectrogram: 2D array [224, 224] or None
    """
    if spectrogram is None:
        # Show placeholder
        st.info("No inference data yet")
        return

    fig = go.Figure(
        data=go.Heatmap(
            z=spectrogram, colorscale="Viridis", showscale=False, hoverinfo="skip"
        )
    )

    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False, title="Time"),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            title="Frequency",
            autorange="reversed",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, width="stretch")


def render_probability_bars(probabilities: Optional[List[float]]) -> None:
    """
    Render model output as horizontal probability bars.

    Args:
        probabilities: [p_background, p_vessel, p_cetacean] or None
    """
    st.markdown("**Model Output**")

    if probabilities is None:
        st.caption("No inference results")
        return

    # Create horizontal bar chart
    fig = go.Figure()

    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
        color = CLASS_COLORS.get(name, "#6b7280")
        fig.add_trace(
            go.Bar(
                y=[name],
                x=[prob * 100],
                orientation="h",
                marker=dict(color=color),
                text=f"{prob:.1%}",
                textposition="inside",
                textfont=dict(color="white"),
                showlegend=False,
            )
        )

    fig.update_layout(
        height=120,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            range=[0, 100],
            title="Probability %",
        ),
        yaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
    )

    st.plotly_chart(fig, width="stretch")


def render_prediction_badge(
    class_id: Optional[int], confidence: Optional[float]
) -> None:
    """
    Render prediction badge with class and confidence.

    Args:
        class_id: Predicted class (0=Background, 1=Vessel, 2=Cetacean)
        confidence: Model confidence (0-1)
    """
    if class_id is None:
        st.caption("No prediction")
        return

    class_name = (
        CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else "Unknown"
    )
    color = CLASS_COLORS.get(class_name, "#6b7280")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Prediction**")
        st.markdown(
            f"""<span style="
                display: inline-block;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 18px;
                background-color: {color};
                color: white;
            ">{class_name}</span>""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("**Confidence**")
        conf_pct = confidence * 100 if confidence else 0
        # Color confidence based on threshold
        conf_color = "#10b981" if confidence and confidence >= 0.85 else "#f59e0b"
        st.markdown(
            f"<span style='font-size: 28px; font-weight: 600; color: {conf_color}; "
            f"font-family: JetBrains Mono, monospace;'>{conf_pct:.1f}%</span>",
            unsafe_allow_html=True,
        )


def render_spectrogram_panel(
    current_state: Optional[Dict[str, Any]],
    last_inference_audio: Optional[str],
    last_spectrogram: Optional[np.ndarray],
) -> None:
    """
    Render the spectrogram panel.

    Args:
        current_state: Most recent tick state dict
        last_inference_audio: Base64-encoded audio from last inference
        last_spectrogram: Cached spectrogram array (to avoid recomputation)
    """
    st.markdown("### Spectrogram")

    # Check if we have new inference data
    inference_run = (
        current_state.get("inference_run", False) if current_state else False
    )

    # Generate or use cached spectrogram
    if inference_run and last_inference_audio:
        # New inference - generate fresh spectrogram
        spectrogram = render_spectrogram(last_inference_audio)
        label = "Current inference"
    elif last_spectrogram is not None:
        spectrogram = last_spectrogram
        label = "Previous inference"
    else:
        spectrogram = None
        label = None

    if label:
        st.caption(label)

    # Render spectrogram heatmap
    render_spectrogram_heatmap(spectrogram)

    # Get inference results from current state
    if current_state and current_state.get("inference_run"):
        probabilities = current_state.get("probabilities")
        class_id = current_state.get("class_id")
        confidence = current_state.get("confidence")
    else:
        probabilities = None
        class_id = None
        confidence = None

    # Probability bars and prediction
    col1, col2 = st.columns([2, 1])

    with col1:
        render_probability_bars(probabilities)

    with col2:
        render_prediction_badge(class_id, confidence)
