#!/usr/bin/env python3
"""
Main simulation loop for marine sensor node.

Implements the state machine and tick loop per spec Section 2.
States: LISTENING, INFERENCE (logical), TRANSMIT, SLEEP, SHUTDOWN
"""

import sys
import os
from enum import Enum, auto
from typing import Optional

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    WAKE_THRESHOLD, CONFIDENCE_THRESHOLD,
    TX_DURATION_TICKS, TX_COOLDOWN_TICKS,
    CHUNK_DURATION, MODEL_PATH, SIM_INPUT_DIR, LOG_FILE,
    SAMPLE_RATE, BATTERY_CAPACITY_MAH, CLASS_NAMES
)
from hardware_sim import Battery, AudioEnvironment
from dsp_pipeline import AudioBuffer, compute_rms, normalize
from inference import InferenceEngine
from comms import TelemetryLogger, log_detection, log_shutdown


class State(Enum):
    """Firmware states per spec Section B."""
    LISTENING = auto()
    INFERENCE = auto()  # Logical sub-state within LISTENING tick
    TRANSMIT = auto()
    SLEEP = auto()
    SHUTDOWN = auto()


class MissionStats:
    """Track mission statistics for summary report."""

    def __init__(self):
        self.vessel_count = 0
        self.cetacean_count = 0
        self.inference_runs = 0
        self.blind_ticks = 0  # TRANSMIT + SLEEP
        self.total_ticks = 0


def validate_runtime_config() -> None:
    """
    Validate config values at simulation startup.

    Per spec Section 7.3: Called from main.py at runtime, NOT at import time.
    This allows train_model.py to import config without model file existing.
    """
    assert SAMPLE_RATE > 0, "SAMPLE_RATE must be positive"
    assert 0.0 < WAKE_THRESHOLD < 1.0, "WAKE_THRESHOLD must be in (0, 1)"
    assert 0.0 < CONFIDENCE_THRESHOLD <= 1.0, "CONFIDENCE_THRESHOLD must be in (0, 1]"
    assert TX_DURATION_TICKS > 0, "TX_DURATION_TICKS must be positive"
    assert TX_COOLDOWN_TICKS >= 0, "TX_COOLDOWN_TICKS must be non-negative"
    assert BATTERY_CAPACITY_MAH > 0, "BATTERY_CAPACITY_MAH must be positive"

    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        print("Run train_model.py first to create the model.")
        sys.exit(1)

    if not os.path.isdir(SIM_INPUT_DIR):
        print(f"ERROR: Input directory not found: {SIM_INPUT_DIR}")
        sys.exit(1)

    log_dir = os.path.dirname(LOG_FILE) or '.'
    if not os.access(log_dir, os.W_OK):
        print(f"ERROR: Log directory not writable: {log_dir}", file=sys.stderr)
        sys.exit(1)


def print_mission_summary(
    reason: str,
    tick_count: int,
    battery: Battery,
    stats: MissionStats
) -> None:
    """
    Print mission summary per spec Section 8.

    Args:
        reason: "LOW_BATTERY" or "AUDIO_EXHAUSTED"
        tick_count: Total simulation ticks
        battery: Battery instance for final state
        stats: MissionStats with detection counts
    """
    # Calculate time
    total_seconds = tick_count * CHUNK_DURATION
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    # Calculate coverage
    active_ticks = tick_count - stats.blind_ticks
    active_pct = 100.0 * active_ticks / tick_count if tick_count > 0 else 0
    blind_pct = 100.0 * stats.blind_ticks / tick_count if tick_count > 0 else 0

    # Calculate power
    mah_used = battery.mah_consumed()
    avg_current = mah_used / (total_seconds / 3600) if total_seconds > 0 else 0

    # Format termination reason
    term_reason = "AUDIO_EXHAUSTED" if reason == "AUDIO_EXHAUSTED" else "LOW_BATTERY"

    print("=" * 80)
    print("                            MISSION SUMMARY")
    print("=" * 80)
    print(f"Termination Reason : {term_reason}")
    print(f"Total Runtime      : {tick_count} ticks ({hours}h {minutes}m {seconds}s)")
    print(f"Final Battery      : {battery.voltage:.2f}V ({battery.capacity_percent:.1f}%)")
    print()
    print("DETECTIONS:")
    print(f"  Vessels          : {stats.vessel_count}")
    print(f"  Cetaceans        : {stats.cetacean_count}")
    print(f"  Total Events     : {stats.vessel_count + stats.cetacean_count}")
    print()
    print("COVERAGE:")
    print(f"  Active Time      : {active_ticks} ticks ({active_pct:.1f}%)")
    print(f"  Blind Time       : {stats.blind_ticks} ticks ({blind_pct:.1f}%)")
    print(f"  Inference Runs   : {stats.inference_runs}")
    print()
    print("POWER CONSUMPTION:")
    print(f"  Total mAh Used   : {mah_used:.1f} mAh")
    print(f"  Avg Current      : {avg_current:.1f} mA")
    print("=" * 80)


def run_simulation() -> None:
    """
    Main simulation entry point.

    Implements the tick loop contract per spec Section 2.
    """
    print("=" * 60)
    print("Marine Sensor Node Simulation")
    print("=" * 60)

    # Validate configuration
    validate_runtime_config()

    # Initialize components
    battery = Battery()
    env = AudioEnvironment()
    buffer = AudioBuffer()
    engine = InferenceEngine()
    logger = TelemetryLogger()
    stats = MissionStats()

    # State machine variables
    state = State.LISTENING
    tick_count = 0
    tx_timer = 0
    sleep_timer = 0
    last_rms = 0.0  # Track RMS for logging

    logger.open()

    print(f"\nStarting simulation...")
    print(f"  Battery: {battery.voltage:.2f}V ({battery.capacity_percent:.0f}%)")
    print(f"  Wake threshold: {WAKE_THRESHOLD}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("-" * 60)

    try:
        while True:
            # ============================================
            # TICK LOOP CONTRACT (Exact ordering per spec)
            # ============================================

            # 1. CONSUME POWER for current state (before any state change)
            state_name = state.name
            if state == State.INFERENCE:
                state_name = "LISTENING"  # INFERENCE uses LISTENING power
            battery.consume_power(state_name)

            # 2. UPDATE VOLTAGE after consumption
            battery.update_voltage()

            # 3. CHECK LOW-BATTERY (triggers SHUTDOWN if V <= 3.2)
            if battery.is_low():
                log_shutdown(logger, tick_count, battery.voltage, "LOW_BATTERY")
                logger.flush()
                print(f"\n[Tick {tick_count}] LOW BATTERY SHUTDOWN at {battery.voltage:.2f}V")
                print_mission_summary("LOW_BATTERY", tick_count, battery, stats)
                sys.exit(0)

            # 4. READ AUDIO from environment (or discard if blind)
            if state in (State.TRANSMIT, State.SLEEP):
                # Advance pointer, discard data (blind period)
                _ = env.read_chunk()
                raw_chunk = None
                stats.blind_ticks += 1
            else:
                raw_chunk = env.read_chunk()
                if raw_chunk is None:
                    # Audio exhausted
                    log_shutdown(logger, tick_count, battery.voltage, "AUDIO_EXHAUSTED")
                    logger.flush()
                    print(f"\n[Tick {tick_count}] AUDIO EXHAUSTED - Mission complete")
                    print_mission_summary("AUDIO_EXHAUSTED", tick_count, battery, stats)
                    sys.exit(0)

            # 5. STATE-SPECIFIC LOGIC
            if state == State.LISTENING:
                # Compute RMS on raw audio (before normalization)
                rms_raw = compute_rms(raw_chunk)
                last_rms = rms_raw

                # Normalize and add to buffer
                normalized = normalize(raw_chunk)
                buffer.append(normalized)

                # Check wake condition
                if buffer.is_full() and rms_raw > WAKE_THRESHOLD:
                    state = State.INFERENCE
                    # Fall through to INFERENCE logic THIS tick

            if state == State.INFERENCE:
                # Generate spectrogram and run model
                spectrogram = buffer.get_spectrogram()
                class_id, confidence = engine.run(spectrogram)
                stats.inference_runs += 1

                # Check detection criteria
                if confidence >= CONFIDENCE_THRESHOLD and class_id in (1, 2):
                    # Valid detection
                    log_detection(logger, tick_count, battery.voltage,
                                  class_id, confidence, last_rms)

                    if class_id == 1:
                        stats.vessel_count += 1
                        print(f"[Tick {tick_count}] VESSEL detected "
                              f"(conf={confidence:.2f}, rms={last_rms:.3f})")
                    else:
                        stats.cetacean_count += 1
                        print(f"[Tick {tick_count}] CETACEAN detected "
                              f"(conf={confidence:.2f}, rms={last_rms:.3f})")

                    # Enter TRANSMIT state
                    buffer.clear()
                    tx_timer = TX_DURATION_TICKS
                    state = State.TRANSMIT
                else:
                    # No detection - return to listening
                    state = State.LISTENING

            elif state == State.TRANSMIT:
                tx_timer -= 1
                if tx_timer == 0:
                    sleep_timer = TX_COOLDOWN_TICKS
                    state = State.SLEEP

            elif state == State.SLEEP:
                sleep_timer -= 1
                if sleep_timer == 0:
                    buffer.clear()  # Defensive clear on SLEEP exit
                    state = State.LISTENING

            # 6. INCREMENT TICK COUNTER
            tick_count += 1
            stats.total_ticks = tick_count

            # Progress indicator every 1000 ticks
            if tick_count % 1000 == 0:
                print(f"[Tick {tick_count}] Battery: {battery.voltage:.2f}V "
                      f"({battery.capacity_percent:.1f}%), State: {state.name}")

    except KeyboardInterrupt:
        print(f"\n\nSimulation interrupted at tick {tick_count}")
        logger.flush()
        print_mission_summary("INTERRUPTED", tick_count, battery, stats)
        sys.exit(0)

    finally:
        logger.close()


if __name__ == "__main__":
    run_simulation()
