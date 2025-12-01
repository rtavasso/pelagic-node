"""
Power panel component for dashboard.

Displays battery status, voltage, current draw, and power consumption history.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import BATTERY_CAPACITY_MAH


# Battery gauge colors - professional muted tones
def get_battery_color(capacity_pct: float) -> str:
    """Get battery color based on capacity percentage."""
    if capacity_pct > 50:
        return "#10b981"  # Emerald
    elif capacity_pct > 20:
        return "#f59e0b"  # Amber
    else:
        return "#ef4444"  # Red


def calc_projected_runtime(capacity_pct: float, avg_current_ma: float) -> str:
    """
    Estimate remaining runtime based on recent average current.

    Args:
        capacity_pct: Current battery capacity percentage
        avg_current_ma: Average current draw in mA

    Returns:
        Formatted string like "~5h 30m" or "N/A"
    """
    remaining_mah = (capacity_pct / 100.0) * BATTERY_CAPACITY_MAH
    if avg_current_ma <= 0:
        return "N/A"

    remaining_hours = remaining_mah / avg_current_ma
    remaining_seconds = remaining_hours * 3600

    hours = int(remaining_seconds // 3600)
    minutes = int((remaining_seconds % 3600) // 60)
    return f"~{hours}h {minutes}m"


def render_power_panel(
    current_state: Optional[Dict[str, Any]],
    tick_history: List[Dict[str, Any]],
    rolling_window: int = 300,
) -> None:
    """
    Render the power panel.

    Args:
        current_state: Most recent tick state dict
        tick_history: List of all tick states for history graph
        rolling_window: Number of ticks for rolling average (default 300)
    """
    st.markdown("### Power")

    if not current_state:
        st.info("Waiting for data...")
        return

    voltage = current_state.get("voltage", 0)
    capacity_pct = current_state.get("capacity_pct", 0)
    current_ma = current_state.get("current_ma", 0)

    # Calculate rolling average current
    recent_ticks = tick_history[-rolling_window:] if tick_history else []
    if recent_ticks:
        avg_current_ma = sum(t.get("current_ma", 0) for t in recent_ticks) / len(
            recent_ticks
        )
    else:
        avg_current_ma = current_ma

    projected_runtime = calc_projected_runtime(capacity_pct, avg_current_ma)
    battery_color = get_battery_color(capacity_pct)

    # Layout in columns
    col1, col2 = st.columns(2)

    with col1:
        # Battery gauge
        st.markdown("**Battery**")
        st.progress(capacity_pct / 100.0)
        st.markdown(
            f"<span style='color: {battery_color}; font-size: 24px; font-weight: bold;'>"
            f"{capacity_pct:.1f}%</span>",
            unsafe_allow_html=True,
        )

        # Voltage
        st.markdown("**Voltage**")
        st.markdown(
            f"<span style='font-size: 20px;'>{voltage:.2f}V</span>",
            unsafe_allow_html=True,
        )

    with col2:
        # Current draw
        st.markdown("**Current Draw**")
        st.markdown(
            f"<span style='font-size: 20px;'>{current_ma} mA</span>",
            unsafe_allow_html=True,
        )

        # Projected runtime
        st.markdown("**Projected Runtime**")
        st.markdown(
            f"<span style='font-size: 20px;'>{projected_runtime}</span>",
            unsafe_allow_html=True,
        )

    # Power consumption history graph (rolling window)
    if len(tick_history) > 1:
        st.markdown("**Consumption History**")

        # Get last N ticks for display
        display_ticks = tick_history[-rolling_window:]
        ticks = [t.get("tick", i) for i, t in enumerate(display_ticks)]
        currents = [t.get("current_ma", 0) for t in display_ticks]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ticks,
                y=currents,
                mode="lines",
                fill="tozeroy",
                line=dict(color="#06b6d4", width=1),
                fillcolor="rgba(6, 182, 212, 0.3)",
            )
        )

        fig.update_layout(
            height=150,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)",
                title="mA",
                range=[0, 700],
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, width="stretch")
