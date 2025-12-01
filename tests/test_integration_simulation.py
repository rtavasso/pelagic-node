"""
Integration tests exercising the full simulation loop in a controlled, fast configuration.

These tests patch heavy dependencies and timers to validate end-to-end behavior
without long runtimes or ONNX dependencies.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _build_env_factory(sim_input: Path):
    """Return a factory that builds AudioEnvironment bound to the temp sim_input."""
    from hardware_sim import AudioEnvironment

    def factory():
        return AudioEnvironment(str(sim_input))

    return factory


def _build_logger_factory(log_path: Path):
    """Return a factory that builds TelemetryLogger bound to the temp log path."""
    from comms import TelemetryLogger

    def factory():
        return TelemetryLogger(str(log_path))

    return factory


def test_simulation_audio_exhaustion_single_detection(tmp_path, make_high_rms_wav, stub_inference_engine, monkeypatch):
    """
    Run the simulation to AUDIO_EXHAUSTED with a single detection and ensure logs are written.
    """
    # Paths
    sim_input = tmp_path / "data" / "sim_input"
    sim_input.mkdir(parents=True)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    log_file = logs_dir / "mission_log.jsonl"

    # Audio: 3 seconds of high-RMS tone to trigger a detection once
    make_high_rms_wav(sim_input / "audio.wav", duration_sec=3.0)

    # Stub inference returns vessel with high confidence
    stub_inference_engine.set_response(class_id=1, confidence=0.99)

    # Patch components to use temp paths and stubs
    import main

    monkeypatch.setattr("main.validate_runtime_config", lambda: None)
    monkeypatch.setattr("main.InferenceEngine", lambda *a, **k: stub_inference_engine)
    monkeypatch.setattr("main.AudioEnvironment", _build_env_factory(sim_input))
    monkeypatch.setattr("main.TelemetryLogger", _build_logger_factory(log_file))

    # Shorten timers for speed
    monkeypatch.setattr("main.TX_DURATION_TICKS", 1)
    monkeypatch.setattr("main.TX_COOLDOWN_TICKS", 1)

    # Run simulation and capture SystemExit on AUDIO_EXHAUSTED
    with pytest.raises(SystemExit) as exc_info:
        main.run_simulation()

    assert exc_info.value.code == 0
    assert log_file.exists(), "Telemetry log should be created"

    # Log should contain one detection and one mission_end
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 2, f"Expected detection + mission_end, got {len(lines)}"
    assert "VESSEL" in lines[0], f"First event should be detection, got: {lines[0]}"
    assert "MISSION_END" in lines[1], f"Second event should be mission end, got: {lines[1]}"


def test_simulation_low_battery_shutdown(tmp_path, make_high_rms_wav, stub_inference_engine, monkeypatch):
    """
    Force a low-battery shutdown and ensure shutdown is logged without detections.
    """
    sim_input = tmp_path / "data" / "sim_input"
    sim_input.mkdir(parents=True)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    log_file = logs_dir / "mission_log.jsonl"

    # Minimal audio; we want battery to die quickly
    make_high_rms_wav(sim_input / "audio.wav", duration_sec=1.0)

    # Stub inference returns background to avoid detections
    stub_inference_engine.set_response(class_id=0, confidence=0.1)

    # Custom battery that starts near cutoff
    import hardware_sim

    class LowBattery(hardware_sim.Battery):
        def __init__(self):
            super().__init__()
            # Set capacity so first tick drops below 3.2V
            self.current_capacity_mah = 1666  # ~3.2V after consume_power

    # Patch components
    import main

    monkeypatch.setattr("main.validate_runtime_config", lambda: None)
    monkeypatch.setattr("main.InferenceEngine", lambda *a, **k: stub_inference_engine)
    monkeypatch.setattr("main.AudioEnvironment", _build_env_factory(sim_input))
    monkeypatch.setattr("main.TelemetryLogger", _build_logger_factory(log_file))
    monkeypatch.setattr("main.Battery", LowBattery)
    monkeypatch.setattr("main.TX_DURATION_TICKS", 1)
    monkeypatch.setattr("main.TX_COOLDOWN_TICKS", 1)

    with pytest.raises(SystemExit) as exc_info:
        main.run_simulation()

    assert exc_info.value.code == 0
    assert log_file.exists(), "Telemetry log should be created"

    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 1, f"Expected only shutdown event, got {len(lines)}"
    assert "LOW_BATTERY_SHUTDOWN" in lines[0], f"Shutdown event missing: {lines[0]}"
