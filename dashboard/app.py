"""
Pelagic Node Dashboard - Main Streamlit Application

Replay-only dashboard for the marine acoustic anomaly detection simulator.
In the hosted (Streamlit Cloud) deployment we only read pre-generated log files.
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "utils"))
sys.path.insert(0, str(Path(__file__).parent / "components"))

from audio_codec import render_spectrogram
from audio_panel import render_audio_panel
from detection_panel import render_detection_panel
from event_log import render_event_log
from header import render_header
from log_watcher import LineBufferedLogWatcher, list_archived_missions
from power_panel import render_power_panel
from spectrogram_panel import render_spectrogram_panel
from state_machine import render_state_machine_panel

from config import CHUNK_DURATION, LOG_ARCHIVE_DIR, LOG_FILE, TICK_STATE_FILE


def load_css():
    """Load custom CSS theme."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state with default values."""
    defaults = {
        # Replay-only mode
        "selected_archive": None,
        # Accumulated data
        "tick_history": [],
        "event_history": [],
        # Current display state
        "current_tick": 0,
        "current_state": None,
        "last_inference_audio": None,
        "last_spectrogram": None,
        # Derived metrics
        "vessel_count": 0,
        "cetacean_count": 0,
        # Replay state
        "replay_tick_index": 0,
        "playback_speed": 1.0,
        "is_paused": True,
        "replay_data_loaded": False,
        "replay_last_control": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session_data():
    """Reset session data when switching modes or missions."""
    st.session_state.tick_history = []
    st.session_state.event_history = []
    st.session_state.current_tick = 0
    st.session_state.current_state = None
    st.session_state.last_inference_audio = None
    st.session_state.last_spectrogram = None
    st.session_state.vessel_count = 0
    st.session_state.cetacean_count = 0
    st.session_state.replay_tick_index = 0
    st.session_state.replay_data_loaded = False
    st.session_state.replay_last_control = None
    if "seek_slider" in st.session_state:
        del st.session_state["seek_slider"]


def render_sidebar():
    """Render sidebar with replay controls only."""
    with st.sidebar:
        st.markdown("## Controls")
        st.info("Replay-only: upload or commit log files, no live simulation required.")

        # Build mission list: current logs (if present) + archived runs
        archives = list_archived_missions()
        mission_options = []
        if Path(TICK_STATE_FILE).exists() and Path(LOG_FILE).exists():
            mission_options.append("current")
        mission_options.extend(archives)

        if not mission_options:
            st.warning(
                "No mission logs found. Add files to ./logs or ./logs/archive/mission_*/"
            )
            return

        # Auto-select latest if nothing chosen
        if st.session_state.selected_archive not in mission_options:
            st.session_state.selected_archive = mission_options[0]

        def _label(option: str) -> str:
            if option == "current":
                return "Current logs (./logs)"
            return f"Archive: {option}"

        selected = st.selectbox(
            "Select Mission",
            mission_options,
            index=mission_options.index(st.session_state.selected_archive),
            format_func=_label,
            key="archive_selector",
        )

        if selected != st.session_state.selected_archive:
            st.session_state.selected_archive = selected
            reset_session_data()

        # Playback controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Step Back", key="step_back"):
                if st.session_state.replay_tick_index > 0:
                    st.session_state.replay_tick_index -= 1
                    st.session_state.replay_last_control = "button"

        with col2:
            if st.button("Step Fwd", key="step_fwd"):
                max_idx = len(st.session_state.tick_history) - 1
                if st.session_state.replay_tick_index < max_idx:
                    st.session_state.replay_tick_index += 1
                    st.session_state.replay_last_control = "button"

        # Play/Pause
        if st.session_state.is_paused:
            if st.button("Play", key="play_btn", use_container_width=True):
                st.session_state.is_paused = False
        else:
            if st.button("Pause", key="pause_btn", use_container_width=True):
                st.session_state.is_paused = True

        # Speed control
        speed = st.select_slider(
            "Speed",
            options=[1, 10, 100],
            value=int(st.session_state.playback_speed),
            key="speed_slider",
        )
        st.session_state.playback_speed = float(speed)

        # Seek slider with two-way sync to replay_tick_index
        if st.session_state.tick_history:
            max_tick = len(st.session_state.tick_history) - 1
            # Keep slider in sync when index is changed programmatically
            if "seek_slider" not in st.session_state:
                st.session_state.seek_slider = st.session_state.replay_tick_index
            elif st.session_state.get("replay_last_control") in ("button", "auto"):
                st.session_state.seek_slider = st.session_state.replay_tick_index

            seek_pos = st.slider(
                "Seek",
                0,
                max_tick,
                key="seek_slider",
            )

            if seek_pos != st.session_state.replay_tick_index:
                st.session_state.replay_tick_index = seek_pos
                st.session_state.replay_last_control = "slider"

        st.divider()

        # Info section
        st.markdown("### Info")
        st.markdown(f"**Tick Duration:** {CHUNK_DURATION}s")

        if st.session_state.tick_history:
            st.markdown(f"**Total Ticks:** {len(st.session_state.tick_history):,}")
            st.markdown(f"**Vessels:** {st.session_state.vessel_count}")
            st.markdown(f"**Cetaceans:** {st.session_state.cetacean_count}")


def run_replay_mode():
    """Main loop for replay mode."""
    if not st.session_state.selected_archive:
        st.info("Select an archived mission from the sidebar")
        return

    # Load replay data if not loaded
    if not st.session_state.replay_data_loaded:
        if st.session_state.selected_archive == "current":
            tick_path = Path(TICK_STATE_FILE)
            event_path = Path(LOG_FILE)
        else:
            archive_path = Path(LOG_ARCHIVE_DIR) / st.session_state.selected_archive
            tick_path = archive_path / "tick_state.jsonl"
            event_path = archive_path / "mission_log.jsonl"

        if tick_path.exists():
            tick_watcher = LineBufferedLogWatcher(str(tick_path))
            st.session_state.tick_history = tick_watcher.read_all()

        if event_path.exists():
            event_watcher = LineBufferedLogWatcher(str(event_path))
            st.session_state.event_history = event_watcher.read_all()

            # Count events
            for event in st.session_state.event_history:
                if event.get("event") == "VESSEL":
                    st.session_state.vessel_count += 1
                elif event.get("event") == "CETACEAN":
                    st.session_state.cetacean_count += 1

        st.session_state.replay_data_loaded = True

    if not st.session_state.tick_history:
        st.warning("No tick data found in archive")
        return

    # Clamp index in case previous selection had more ticks
    max_idx = len(st.session_state.tick_history) - 1
    if st.session_state.replay_tick_index > max_idx:
        st.session_state.replay_tick_index = max_idx

    # Get current tick state for display
    idx = st.session_state.replay_tick_index
    current_state = st.session_state.tick_history[idx]
    st.session_state.current_state = current_state

    # Update audio cache if this tick had inference
    if current_state.get("inference_run") and current_state.get("audio_b64gz"):
        st.session_state.last_inference_audio = current_state["audio_b64gz"]
        spec = render_spectrogram(current_state["audio_b64gz"])
        if spec is not None:
            st.session_state.last_spectrogram = spec

    # Get history up to current tick for panels that need it
    history_to_current = st.session_state.tick_history[: idx + 1]

    # Filter events up to current tick
    current_tick = current_state.get("tick", 0)
    events_to_current = [
        e for e in st.session_state.event_history if e.get("ts", 0) <= current_tick
    ]

    # Render all panels
    render_header(current_state, "COMPLETED")

    # Top row
    col1, col2 = st.columns(2)
    with col1:
        render_power_panel(current_state, history_to_current)
    with col2:
        vessel_count = sum(1 for e in events_to_current if e.get("event") == "VESSEL")
        cetacean_count = sum(
            1 for e in events_to_current if e.get("event") == "CETACEAN"
        )
        render_detection_panel(events_to_current, vessel_count, cetacean_count)

    # State machine
    render_state_machine_panel(current_state, history_to_current)

    # Bottom row
    col1, col2 = st.columns(2)
    with col1:
        render_audio_panel(current_state, st.session_state.last_inference_audio)
    with col2:
        render_spectrogram_panel(
            current_state,
            st.session_state.last_inference_audio,
            st.session_state.last_spectrogram,
        )

    # Event log
    render_event_log(events_to_current)

    # Auto-advance if playing
    if not st.session_state.is_paused:
        max_idx = len(st.session_state.tick_history) - 1
        if st.session_state.replay_tick_index < max_idx:
            delay = CHUNK_DURATION / st.session_state.playback_speed
            time.sleep(delay)
            st.session_state.replay_tick_index += 1
            st.session_state.replay_last_control = "auto"
            st.rerun()
        else:
            # Reached end, pause
            st.session_state.is_paused = True


def main():
    """Main entry point."""
    st.set_page_config(
        page_title="Pelagic Node Monitor",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_css()
    init_session_state()
    render_sidebar()

    run_replay_mode()


if __name__ == "__main__":
    main()
