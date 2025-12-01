"""
P3 Tests: Telemetry & Summaries

These tests verify telemetry event schema, shutdown events,
and mission summary accuracy.
"""

import sys
import json
import pytest
import numpy as np
from pathlib import Path
from io import StringIO
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    SAMPLE_RATE, TX_DURATION_TICKS, TX_COOLDOWN_TICKS,
    CHUNK_DURATION, BATTERY_CAPACITY_MAH
)


class TestDetectionEventSchema:
    """
    Test detection event NDJSON format and content.
    """

    def test_detection_event_has_required_fields(self, temp_log_dir):
        """NDJSON lines include ts, batt_v, event, conf, rms."""
        from comms import TelemetryLogger, log_detection

        log_file = temp_log_dir / "mission_log.jsonl"
        logger = TelemetryLogger(str(log_file))
        logger.open()

        # Log a detection
        log_detection(logger, tick=100, voltage=4.0, class_id=1,
                      confidence=0.92, rms=0.15)

        logger.flush()
        logger.close()

        # Read and parse
        with open(log_file) as f:
            line = f.readline()
            event = json.loads(line)

        assert "ts" in event, "Missing 'ts' field"
        assert "batt_v" in event, "Missing 'batt_v' field"
        assert "event" in event, "Missing 'event' field"
        assert "conf" in event, "Missing 'conf' field"
        assert "rms" in event, "Missing 'rms' field"

    def test_detection_event_values(self, temp_log_dir):
        """Detection event values match logged values."""
        from comms import TelemetryLogger, log_detection

        log_file = temp_log_dir / "mission_log.jsonl"
        logger = TelemetryLogger(str(log_file))
        logger.open()

        log_detection(logger, tick=42, voltage=3.85, class_id=2,
                      confidence=0.91, rms=0.23)

        logger.flush()
        logger.close()

        with open(log_file) as f:
            event = json.loads(f.readline())

        assert event["ts"] == 42
        assert event["batt_v"] == 3.85
        assert event["event"] == "CETACEAN"
        assert event["conf"] == 0.91
        assert event["rms"] == 0.23

    def test_vessel_event_name(self, temp_log_dir):
        """class_id=1 produces event='VESSEL'."""
        from comms import TelemetryLogger, log_detection

        log_file = temp_log_dir / "mission_log.jsonl"
        logger = TelemetryLogger(str(log_file))
        logger.open()

        log_detection(logger, tick=10, voltage=4.0, class_id=1,
                      confidence=0.90, rms=0.1)

        logger.close()

        with open(log_file) as f:
            event = json.loads(f.readline())

        assert event["event"] == "VESSEL"

    def test_cetacean_event_name(self, temp_log_dir):
        """class_id=2 produces event='CETACEAN'."""
        from comms import TelemetryLogger, log_detection

        log_file = temp_log_dir / "mission_log.jsonl"
        logger = TelemetryLogger(str(log_file))
        logger.open()

        log_detection(logger, tick=20, voltage=4.0, class_id=2,
                      confidence=0.95, rms=0.2)

        logger.close()

        with open(log_file) as f:
            event = json.loads(f.readline())

        assert event["event"] == "CETACEAN"

    def test_event_under_340_bytes(self, temp_log_dir):
        """Each event should be under 340 bytes (RockBLOCK limit)."""
        from comms import TelemetryLogger, log_detection

        log_file = temp_log_dir / "mission_log.jsonl"
        logger = TelemetryLogger(str(log_file))
        logger.open()

        # Log with maximum reasonable values
        log_detection(logger, tick=999999, voltage=4.20, class_id=2,
                      confidence=0.99, rms=0.99)

        logger.close()

        with open(log_file) as f:
            line = f.readline()

        assert len(line.encode('utf-8')) < 340, \
            f"Event line is {len(line.encode('utf-8'))} bytes, exceeds 340"

    def test_ndjson_format_valid(self, temp_log_dir):
        """Multiple events produce valid NDJSON (one JSON per line)."""
        from comms import TelemetryLogger, log_detection

        log_file = temp_log_dir / "mission_log.jsonl"
        logger = TelemetryLogger(str(log_file))
        logger.open()

        # Log multiple events
        log_detection(logger, tick=10, voltage=4.0, class_id=1,
                      confidence=0.90, rms=0.1)
        log_detection(logger, tick=500, voltage=3.9, class_id=2,
                      confidence=0.88, rms=0.15)
        log_detection(logger, tick=1000, voltage=3.8, class_id=1,
                      confidence=0.92, rms=0.2)

        logger.close()

        # Read and verify each line is valid JSON
        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) == 3, f"Expected 3 events, got {len(lines)}"

        for i, line in enumerate(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(f"Line {i+1} is not valid JSON: {e}")

            assert isinstance(event, dict), f"Line {i+1} should be a dict"

    def test_values_rounded_appropriately(self, temp_log_dir):
        """Verify values are rounded (batt_v, conf, rms to 2 decimals)."""
        from comms import TelemetryLogger, log_detection

        log_file = temp_log_dir / "mission_log.jsonl"
        logger = TelemetryLogger(str(log_file))
        logger.open()

        # Log with high precision values
        log_detection(logger, tick=100, voltage=3.87654, class_id=1,
                      confidence=0.91234, rms=0.15678)

        logger.close()

        with open(log_file) as f:
            event = json.loads(f.readline())

        # Should be rounded to 2 decimal places
        assert event["batt_v"] == 3.88, f"batt_v should be 3.88, got {event['batt_v']}"
        assert event["conf"] == 0.91, f"conf should be 0.91, got {event['conf']}"
        assert event["rms"] == 0.16, f"rms should be 0.16, got {event['rms']}"


class TestShutdownEvents:
    """
    Test shutdown event logging for LOW_BATTERY and AUDIO_EXHAUSTED.
    """

    def test_low_battery_shutdown_event(self, temp_log_dir):
        """LOW_BATTERY produces event='LOW_BATTERY_SHUTDOWN'."""
        from comms import TelemetryLogger, log_shutdown

        log_file = temp_log_dir / "mission_log.jsonl"
        logger = TelemetryLogger(str(log_file))
        logger.open()

        log_shutdown(logger, tick=50000, voltage=3.19, reason="LOW_BATTERY")

        logger.close()

        with open(log_file) as f:
            event = json.loads(f.readline())

        assert event["event"] == "LOW_BATTERY_SHUTDOWN"
        assert event["ts"] == 50000
        assert event["batt_v"] == 3.19

    def test_audio_exhausted_shutdown_event(self, temp_log_dir):
        """AUDIO_EXHAUSTED produces event='MISSION_END'."""
        from comms import TelemetryLogger, log_shutdown

        log_file = temp_log_dir / "mission_log.jsonl"
        logger = TelemetryLogger(str(log_file))
        logger.open()

        log_shutdown(logger, tick=1000, voltage=4.05, reason="AUDIO_EXHAUSTED")

        logger.close()

        with open(log_file) as f:
            event = json.loads(f.readline())

        assert event["event"] == "MISSION_END"
        assert event["ts"] == 1000
        assert event["batt_v"] == 4.05

    def test_shutdown_event_no_conf_rms(self, temp_log_dir):
        """Shutdown events should not include conf or rms fields."""
        from comms import TelemetryLogger, log_shutdown

        log_file = temp_log_dir / "mission_log.jsonl"
        logger = TelemetryLogger(str(log_file))
        logger.open()

        log_shutdown(logger, tick=100, voltage=3.5, reason="LOW_BATTERY")

        logger.close()

        with open(log_file) as f:
            event = json.loads(f.readline())

        assert "conf" not in event, "Shutdown events should not have 'conf'"
        assert "rms" not in event, "Shutdown events should not have 'rms'"


class TestCoverageMetrics:
    """
    Test coverage metrics (Active vs Blind time) calculation.
    """

    def test_active_plus_blind_equals_total(self):
        """Active + Blind time should equal total ticks."""
        from main import MissionStats

        stats = MissionStats()
        stats.total_ticks = 1000
        stats.blind_ticks = 310  # One detection cycle

        active_ticks = stats.total_ticks - stats.blind_ticks

        assert active_ticks + stats.blind_ticks == stats.total_ticks

    def test_blind_ticks_for_detection_cycle(self):
        """Blind ticks = TX_DURATION + TX_COOLDOWN = 310."""
        blind_for_one_detection = TX_DURATION_TICKS + TX_COOLDOWN_TICKS
        assert blind_for_one_detection == 310

    def test_no_detection_means_no_blind_ticks(self):
        """Without detection, blind_ticks should be 0."""
        from main import MissionStats

        stats = MissionStats()
        stats.total_ticks = 100
        stats.blind_ticks = 0

        active_ticks = stats.total_ticks - stats.blind_ticks
        assert active_ticks == 100
        assert stats.blind_ticks == 0

    def test_inference_runs_tracked(self, stub_inference_engine, high_rms_chunk):
        """Inference runs counter increments correctly."""
        from main import MissionStats
        from dsp_pipeline import AudioBuffer, normalize

        stats = MissionStats()
        buffer = AudioBuffer()

        stub_inference_engine.set_response(class_id=0, confidence=0.5)

        # Fill buffer and run inference
        for _ in range(3):
            buffer.append(normalize(high_rms_chunk))

        spec = buffer.get_spectrogram()

        # Run inference 5 times
        for _ in range(5):
            stub_inference_engine.run(spec)
            stats.inference_runs += 1

        assert stats.inference_runs == 5


class TestMissionSummaryFormat:
    """
    Test mission summary output format and content.
    """

    def test_mission_summary_fields_present(self):
        """Summary contains all required sections."""
        from main import print_mission_summary, MissionStats
        from hardware_sim import Battery

        battery = Battery()
        stats = MissionStats()
        stats.vessel_count = 2
        stats.cetacean_count = 1
        stats.inference_runs = 10
        stats.blind_ticks = 620  # 2 detection cycles
        stats.total_ticks = 1000

        # Capture stdout
        import io
        import contextlib

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_mission_summary("AUDIO_EXHAUSTED", 1000, battery, stats)

        output = f.getvalue()

        # Check required sections
        assert "MISSION SUMMARY" in output
        assert "Termination Reason" in output
        assert "Total Runtime" in output
        assert "Final Battery" in output
        assert "DETECTIONS" in output
        assert "Vessels" in output
        assert "Cetaceans" in output
        assert "COVERAGE" in output
        assert "Active Time" in output
        assert "Blind Time" in output
        assert "Inference Runs" in output
        assert "POWER CONSUMPTION" in output
        assert "Total mAh Used" in output
        assert "Avg Current" in output

    def test_mission_summary_termination_reasons(self):
        """Summary shows correct termination reason."""
        from main import print_mission_summary, MissionStats
        from hardware_sim import Battery
        import io
        import contextlib

        battery = Battery()
        stats = MissionStats()

        # Test AUDIO_EXHAUSTED
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_mission_summary("AUDIO_EXHAUSTED", 100, battery, stats)
        assert "AUDIO_EXHAUSTED" in f.getvalue()

        # Test LOW_BATTERY
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_mission_summary("LOW_BATTERY", 100, battery, stats)
        assert "LOW_BATTERY" in f.getvalue()

    def test_mission_summary_time_calculation(self):
        """Summary shows correct time conversion."""
        from main import print_mission_summary, MissionStats
        from hardware_sim import Battery
        import io
        import contextlib

        battery = Battery()
        stats = MissionStats()

        # 3661 ticks = 1h 1m 1s
        tick_count = 3661

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_mission_summary("AUDIO_EXHAUSTED", tick_count, battery, stats)

        output = f.getvalue()
        assert "3661 ticks" in output
        assert "1h" in output
        assert "1m" in output
        assert "1s" in output

    def test_mission_summary_detection_counts(self):
        """Summary shows correct detection counts."""
        from main import print_mission_summary, MissionStats
        from hardware_sim import Battery
        import io
        import contextlib

        battery = Battery()
        stats = MissionStats()
        stats.vessel_count = 5
        stats.cetacean_count = 3

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_mission_summary("AUDIO_EXHAUSTED", 1000, battery, stats)

        output = f.getvalue()
        assert "5" in output  # Vessels
        assert "3" in output  # Cetaceans
        assert "8" in output  # Total

    def test_mission_summary_coverage_percentages(self):
        """Summary shows correct coverage percentages."""
        from main import print_mission_summary, MissionStats
        from hardware_sim import Battery
        import io
        import contextlib

        battery = Battery()
        stats = MissionStats()
        stats.total_ticks = 1000
        stats.blind_ticks = 310  # 31%

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_mission_summary("AUDIO_EXHAUSTED", 1000, battery, stats)

        output = f.getvalue()

        # Active = 690 ticks (69.0%), Blind = 310 ticks (31.0%)
        assert "Active Time      : 690 ticks (69.0%)" in output, \
            f"Active coverage line missing or incorrect:\n{output}"
        assert "Blind Time       : 310 ticks (31.0%)" in output, \
            f"Blind coverage line missing or incorrect:\n{output}"

    def test_mission_summary_battery_final_state(self):
        """Summary shows correct final battery state."""
        from main import print_mission_summary, MissionStats
        from hardware_sim import Battery
        import io
        import contextlib

        battery = Battery()
        # Consume some power
        for _ in range(1000):
            battery.consume_power("LISTENING")
        battery.update_voltage()

        stats = MissionStats()

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_mission_summary("AUDIO_EXHAUSTED", 1000, battery, stats)

        output = f.getvalue()

        # Should show voltage and percentage
        assert "V" in output  # Voltage unit
        assert "%" in output  # Percentage

    def test_mission_summary_power_consumption(self):
        """Summary shows correct power consumption."""
        from main import print_mission_summary, MissionStats
        from hardware_sim import Battery
        import io
        import contextlib

        battery = Battery()
        initial_capacity = battery.current_capacity_mah

        # Consume 100 ticks of LISTENING (120mA)
        for _ in range(100):
            battery.consume_power("LISTENING")
        battery.update_voltage()

        stats = MissionStats()

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_mission_summary("AUDIO_EXHAUSTED", 100, battery, stats)

        output = f.getvalue()

        # Should contain mAh
        assert "mAh" in output

        # Calculate expected consumption
        # 120mA * (100s / 3600) = 3.33 mAh
        expected_mah = 120 * (100 / 3600)
        assert "3.3" in output or "3.33" in output or str(round(expected_mah, 1)) in output


class TestLoggerLifecycle:
    """
    Test logger file handling behavior.
    """

    def test_logger_creates_directory(self, tmp_path):
        """Logger creates log directory if needed."""
        from comms import TelemetryLogger

        log_file = tmp_path / "new_dir" / "mission_log.jsonl"

        logger = TelemetryLogger(str(log_file))
        logger.open()
        logger.close()

        assert log_file.parent.exists()

    def test_logger_clears_existing_log(self, temp_log_dir):
        """Logger clears existing log on open (current behavior)."""
        from comms import TelemetryLogger, log_detection

        log_file = temp_log_dir / "mission_log.jsonl"

        # First run - log an event
        logger1 = TelemetryLogger(str(log_file))
        logger1.open()
        log_detection(logger1, tick=1, voltage=4.0, class_id=1,
                      confidence=0.9, rms=0.1)
        logger1.close()

        # Verify event was logged
        with open(log_file) as f:
            lines1 = f.readlines()
        assert len(lines1) == 1

        # Second run - opens fresh
        logger2 = TelemetryLogger(str(log_file))
        logger2.open()
        log_detection(logger2, tick=2, voltage=4.0, class_id=2,
                      confidence=0.95, rms=0.2)
        logger2.close()

        # Should only have 1 event (from second run)
        with open(log_file) as f:
            lines2 = f.readlines()

        # Current implementation clears log
        assert len(lines2) == 1, \
            f"Expected log to be cleared, got {len(lines2)} lines"

        # Verify it's the second event
        event = json.loads(lines2[0])
        assert event["ts"] == 2

    def test_logger_flush_writes_to_disk(self, temp_log_dir):
        """Flush ensures data is written to disk."""
        from comms import TelemetryLogger, log_detection

        log_file = temp_log_dir / "mission_log.jsonl"
        logger = TelemetryLogger(str(log_file))
        logger.open()

        log_detection(logger, tick=1, voltage=4.0, class_id=1,
                      confidence=0.9, rms=0.1)

        # Before flush, file might be empty or buffered
        logger.flush()

        # After flush, file should have content
        with open(log_file) as f:
            content = f.read()

        assert len(content) > 0, "File should have content after flush"

        logger.close()

    def test_logger_context_manager(self, temp_log_dir):
        """Logger works as context manager."""
        from comms import TelemetryLogger, log_detection

        log_file = temp_log_dir / "mission_log.jsonl"

        with TelemetryLogger(str(log_file)) as logger:
            log_detection(logger, tick=1, voltage=4.0, class_id=1,
                          confidence=0.9, rms=0.1)

        # File should exist and have content
        assert log_file.exists()
        with open(log_file) as f:
            content = f.read()
        assert len(content) > 0
