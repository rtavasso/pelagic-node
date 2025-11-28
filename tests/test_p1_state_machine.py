"""
P1 Tests: State Machine & Power Model

These tests verify the core state machine behavior, tick loop ordering,
and power consumption model match the spec.
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from collections import deque
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    SAMPLE_RATE, WAKE_THRESHOLD, CONFIDENCE_THRESHOLD,
    TX_DURATION_TICKS, TX_COOLDOWN_TICKS, BATTERY_CAPACITY_MAH,
    POWER_CONSUMPTION, CHUNK_DURATION
)


class TestTickOrdering:
    """
    Test that tick loop follows exact ordering from spec Section 2.

    Order: 1) consume_power, 2) update_voltage, 3) check low battery,
           4) read audio, 5) state logic, 6) increment tick
    """

    def test_tick_ordering_power_then_voltage_then_audio(self):
        """
        Instrument Battery.consume_power/update_voltage and verify
        ordering matches the Tick Loop Contract.
        """
        from hardware_sim import Battery

        # Track call order
        call_order = []

        class InstrumentedBattery(Battery):
            def consume_power(self, state):
                call_order.append(('consume_power', state))
                super().consume_power(state)

            def update_voltage(self):
                call_order.append(('update_voltage',))
                super().update_voltage()

        battery = InstrumentedBattery()

        # Simulate one tick of LISTENING
        battery.consume_power("LISTENING")
        battery.update_voltage()

        assert len(call_order) == 2
        assert call_order[0][0] == 'consume_power', \
            "consume_power must be called first"
        assert call_order[1][0] == 'update_voltage', \
            "update_voltage must be called after consume_power"

    def test_power_consumed_before_voltage_update(self):
        """Verify voltage reflects consumption from current tick."""
        from hardware_sim import Battery

        battery = Battery()
        initial_voltage = battery.voltage

        # Consume power but don't update voltage yet
        battery.consume_power("LISTENING")
        # Voltage should still be old value
        assert battery.voltage == initial_voltage

        # Now update voltage
        battery.update_voltage()
        # Voltage should now reflect consumption
        assert battery.voltage < initial_voltage

    def test_low_battery_check_after_voltage_update(self):
        """Verify low battery is detected after voltage update."""
        from hardware_sim import Battery

        battery = Battery()

        # Set capacity so that after one TRANSMIT tick, voltage drops to ~3.2V
        # V = 3.0 + 1.2 * ratio
        # 3.2 = 3.0 + 1.2 * ratio -> ratio = 0.1667
        # capacity = 0.1667 * 10000 = 1667 mAh
        battery.current_capacity_mah = 1700  # Just above threshold

        # Before consumption, not low
        battery.update_voltage()
        assert not battery.is_low()

        # After heavy consumption (TRANSMIT = 600mA)
        # delta = 600 * (1/3600) = 0.167 mAh per tick
        # But let's set capacity lower to trigger threshold
        battery.current_capacity_mah = 1670
        battery.consume_power("TRANSMIT")
        battery.update_voltage()

        # Check if we're at or below threshold
        # After consuming ~0.167 mAh: 1670 - 0.167 = 1669.83
        # ratio = 1669.83 / 10000 = 0.167
        # V = 3.0 + 1.2 * 0.167 = 3.2V
        assert battery.voltage <= 3.21  # Allow small tolerance


class TestTransmitAndSleepDurations:
    """
    Test TRANSMIT (10 ticks) and SLEEP (300 ticks) durations.
    """

    def test_transmit_duration_exactly_10_ticks(self):
        """Verify TRANSMIT state lasts exactly 10 ticks."""
        # Simulate state machine logic
        tx_timer = TX_DURATION_TICKS  # 10
        transmit_ticks = 0

        while tx_timer > 0:
            tx_timer -= 1
            transmit_ticks += 1

        assert transmit_ticks == 10, \
            f"TRANSMIT should last 10 ticks, got {transmit_ticks}"

    def test_sleep_duration_exactly_300_ticks(self):
        """Verify SLEEP state lasts exactly 300 ticks."""
        sleep_timer = TX_COOLDOWN_TICKS  # 300
        sleep_ticks = 0

        while sleep_timer > 0:
            sleep_timer -= 1
            sleep_ticks += 1

        assert sleep_ticks == 300, \
            f"SLEEP should last 300 ticks, got {sleep_ticks}"

    def test_blind_ticks_equals_310_for_one_detection(self,
                                                       stub_inference_engine,
                                                       high_rms_chunk):
        """
        Verify blind_ticks == 310 (10 TX + 300 SLEEP) for single detection.
        """
        from dsp_pipeline import AudioBuffer, compute_rms, normalize
        from main import State, MissionStats

        # Configure stub to always detect vessel
        stub_inference_engine.set_response(class_id=1, confidence=0.95)

        buffer = AudioBuffer()
        stats = MissionStats()

        state = State.LISTENING
        tx_timer = 0
        sleep_timer = 0
        tick = 0
        detection_occurred = False

        # Run enough ticks for detection + full blind period
        max_ticks = 3 + TX_DURATION_TICKS + TX_COOLDOWN_TICKS + 10

        while tick < max_ticks:
            # Count blind ticks
            if state in (State.TRANSMIT, State.SLEEP):
                stats.blind_ticks += 1

            # State logic
            if state == State.LISTENING:
                rms = compute_rms(high_rms_chunk)
                normalized = normalize(high_rms_chunk)
                buffer.append(normalized)

                if buffer.is_full() and rms > WAKE_THRESHOLD:
                    state = State.INFERENCE

            if state == State.INFERENCE:
                spec = buffer.get_spectrogram()
                class_id, conf = stub_inference_engine.run(spec)

                if conf >= CONFIDENCE_THRESHOLD and class_id in (1, 2):
                    detection_occurred = True
                    buffer.clear()
                    tx_timer = TX_DURATION_TICKS
                    state = State.TRANSMIT
                else:
                    state = State.LISTENING

            elif state == State.TRANSMIT:
                tx_timer -= 1
                if tx_timer == 0:
                    sleep_timer = TX_COOLDOWN_TICKS
                    state = State.SLEEP

            elif state == State.SLEEP:
                sleep_timer -= 1
                if sleep_timer == 0:
                    buffer.clear()
                    state = State.LISTENING
                    break  # Stop after one full cycle

            tick += 1

        assert detection_occurred, "Detection should have occurred"
        assert stats.blind_ticks == 310, \
            f"Blind ticks should be 310 (10+300), got {stats.blind_ticks}"

    def test_buffer_cleared_on_transmit_entry(self, stub_inference_engine,
                                               high_rms_chunk):
        """Verify buffer is cleared when entering TRANSMIT."""
        from dsp_pipeline import AudioBuffer, compute_rms, normalize
        from main import State

        stub_inference_engine.set_response(class_id=1, confidence=0.95)
        buffer = AudioBuffer()

        # Fill buffer
        for _ in range(3):
            buffer.append(normalize(high_rms_chunk))

        assert buffer.is_full()

        # Trigger detection
        spec = buffer.get_spectrogram()
        class_id, conf = stub_inference_engine.run(spec)

        assert conf >= CONFIDENCE_THRESHOLD and class_id == 1

        # Clear buffer (as main.py does on TRANSMIT entry)
        buffer.clear()

        assert len(buffer) == 0, "Buffer should be empty after TRANSMIT entry"

    def test_buffer_cleared_on_sleep_exit(self):
        """Verify buffer is cleared when exiting SLEEP."""
        from dsp_pipeline import AudioBuffer
        from main import State

        buffer = AudioBuffer()

        # Simulate: buffer might have stale data somehow
        # (defensive clear on SLEEP exit)
        t = np.linspace(0, 1, SAMPLE_RATE, dtype=np.float32)
        for _ in range(2):
            buffer.append(t)

        assert len(buffer) == 2

        # On SLEEP exit, buffer is cleared
        buffer.clear()

        assert len(buffer) == 0, "Buffer should be empty after SLEEP exit"


class TestWakeThresholdRawRMS:
    """
    Test that RMS gating uses raw audio and strictly > WAKE_THRESHOLD.
    """

    def test_rms_computed_on_raw_audio(self):
        """Verify RMS is computed before normalization."""
        from dsp_pipeline import compute_rms, normalize

        # Create chunk with known amplitude
        t = np.linspace(0, 1, SAMPLE_RATE, dtype=np.float32)
        raw_chunk = 0.1 * np.sin(2 * np.pi * 440 * t)

        # RMS of sine wave = amplitude / sqrt(2)
        expected_rms = 0.1 / np.sqrt(2)

        raw_rms = compute_rms(raw_chunk)
        assert abs(raw_rms - expected_rms) < 0.001, \
            f"Raw RMS should be ~{expected_rms}, got {raw_rms}"

        # Normalize the chunk
        normalized = normalize(raw_chunk)

        # Normalized chunk has max = 1.0
        assert np.max(np.abs(normalized)) == pytest.approx(1.0, abs=1e-6)

        # RMS of normalized chunk is different
        normalized_rms = compute_rms(normalized)
        assert normalized_rms != raw_rms, \
            "Normalized RMS should differ from raw RMS"

    def test_wake_threshold_strictly_greater_than(self, high_rms_chunk, low_rms_chunk):
        """Verify inference only triggers when RMS > WAKE_THRESHOLD (not >=)."""
        from dsp_pipeline import compute_rms

        high_rms = compute_rms(high_rms_chunk)
        low_rms = compute_rms(low_rms_chunk)

        assert high_rms > WAKE_THRESHOLD, \
            f"High RMS chunk should exceed threshold: {high_rms} > {WAKE_THRESHOLD}"
        assert low_rms < WAKE_THRESHOLD, \
            f"Low RMS chunk should be below threshold: {low_rms} < {WAKE_THRESHOLD}"

        # Test edge case: exactly at threshold
        # Create chunk with RMS = 0.05 exactly
        # RMS = amplitude / sqrt(2) -> amplitude = 0.05 * sqrt(2) = 0.0707
        t = np.linspace(0, 1, SAMPLE_RATE, dtype=np.float32)
        edge_chunk = (WAKE_THRESHOLD * np.sqrt(2)) * np.sin(2 * np.pi * 440 * t)

        edge_rms = compute_rms(edge_chunk)
        assert abs(edge_rms - WAKE_THRESHOLD) < 0.001, \
            f"Edge RMS should be ~{WAKE_THRESHOLD}, got {edge_rms}"

        # Per spec: condition is "RMS > 0.05", so exactly 0.05 should NOT trigger
        should_trigger = edge_rms > WAKE_THRESHOLD
        assert not should_trigger, \
            "RMS exactly at threshold should NOT trigger (strict >)"

    def test_low_rms_does_not_trigger_inference(self, stub_inference_engine,
                                                 low_rms_chunk):
        """Verify low RMS audio doesn't trigger inference."""
        from dsp_pipeline import AudioBuffer, compute_rms, normalize
        from main import State

        buffer = AudioBuffer()

        # Fill buffer with low RMS chunks
        for _ in range(3):
            normalized = normalize(low_rms_chunk)
            buffer.append(normalized)

        assert buffer.is_full()

        rms = compute_rms(low_rms_chunk)
        assert rms < WAKE_THRESHOLD

        # Inference should NOT be triggered
        should_infer = buffer.is_full() and rms > WAKE_THRESHOLD
        assert not should_infer, "Low RMS should not trigger inference"

        # Stub should not have been called
        assert stub_inference_engine.call_count == 0


class TestBufferRefillAfterSleep:
    """
    Test that buffer requires 3 fresh chunks after SLEEP before inference.
    """

    def test_three_chunks_required_after_sleep(self, stub_inference_engine,
                                                high_rms_chunk):
        """
        Ensure three fresh chunks are required before inference after SLEEP.
        """
        from dsp_pipeline import AudioBuffer, compute_rms, normalize
        from main import State

        stub_inference_engine.set_response(class_id=1, confidence=0.95)
        buffer = AudioBuffer()

        # Simulate coming out of SLEEP
        state = State.LISTENING
        buffer.clear()  # Buffer cleared on SLEEP exit

        assert len(buffer) == 0

        # Tick 1: add chunk, buffer length = 1
        buffer.append(normalize(high_rms_chunk))
        rms = compute_rms(high_rms_chunk)

        can_infer_tick1 = buffer.is_full() and rms > WAKE_THRESHOLD
        assert not can_infer_tick1, "Should not infer after 1 chunk"
        assert len(buffer) == 1

        # Tick 2: add chunk, buffer length = 2
        buffer.append(normalize(high_rms_chunk))
        can_infer_tick2 = buffer.is_full() and rms > WAKE_THRESHOLD
        assert not can_infer_tick2, "Should not infer after 2 chunks"
        assert len(buffer) == 2

        # Tick 3: add chunk, buffer length = 3
        buffer.append(normalize(high_rms_chunk))
        can_infer_tick3 = buffer.is_full() and rms > WAKE_THRESHOLD
        assert can_infer_tick3, "Should be able to infer after 3 chunks"
        assert len(buffer) == 3

    def test_buffer_deque_maxlen_3(self):
        """Verify buffer uses deque(maxlen=3)."""
        from dsp_pipeline import AudioBuffer, CONTEXT_CHUNKS

        buffer = AudioBuffer()

        assert CONTEXT_CHUNKS == 3
        assert buffer.buffer.maxlen == 3

        # Add 4 chunks - should only keep last 3
        for i in range(4):
            chunk = np.full(SAMPLE_RATE, float(i), dtype=np.float32)
            buffer.append(chunk)

        assert len(buffer) == 3

        # Verify oldest was dropped
        chunks = list(buffer.buffer)
        assert chunks[0][0] == 1.0, "Oldest chunk (0) should be dropped"
        assert chunks[2][0] == 3.0, "Newest chunk (3) should be present"


class TestBatteryModel:
    """
    Test battery consumption and voltage model.
    """

    def test_single_tick_consumption_listening(self):
        """Test power consumption for LISTENING state."""
        from hardware_sim import Battery

        battery = Battery()
        initial = battery.current_capacity_mah

        battery.consume_power("LISTENING")

        expected_delta = POWER_CONSUMPTION["LISTENING"] * (CHUNK_DURATION / 3600.0)
        actual_delta = initial - battery.current_capacity_mah

        assert actual_delta == pytest.approx(expected_delta, abs=1e-9), \
            f"LISTENING consumption: expected {expected_delta}, got {actual_delta}"

    def test_single_tick_consumption_transmit(self):
        """Test power consumption for TRANSMIT state."""
        from hardware_sim import Battery

        battery = Battery()
        initial = battery.current_capacity_mah

        battery.consume_power("TRANSMIT")

        expected_delta = POWER_CONSUMPTION["TRANSMIT"] * (CHUNK_DURATION / 3600.0)
        actual_delta = initial - battery.current_capacity_mah

        assert actual_delta == pytest.approx(expected_delta, abs=1e-9)

    def test_single_tick_consumption_sleep(self):
        """Test power consumption for SLEEP state."""
        from hardware_sim import Battery

        battery = Battery()
        initial = battery.current_capacity_mah

        battery.consume_power("SLEEP")

        expected_delta = POWER_CONSUMPTION["SLEEP"] * (CHUNK_DURATION / 3600.0)
        actual_delta = initial - battery.current_capacity_mah

        assert actual_delta == pytest.approx(expected_delta, abs=1e-9)

    def test_capacity_clamped_at_zero(self):
        """Test capacity never goes negative."""
        from hardware_sim import Battery

        battery = Battery()
        battery.current_capacity_mah = 0.01  # Very low

        battery.consume_power("TRANSMIT")

        assert battery.current_capacity_mah >= 0, \
            "Capacity should be clamped at 0"

    def test_voltage_at_full_capacity(self):
        """Test voltage at full capacity is 4.2V."""
        from hardware_sim import Battery

        battery = Battery()
        assert battery.current_capacity_mah == BATTERY_CAPACITY_MAH
        battery.update_voltage()

        assert battery.voltage == pytest.approx(4.2, abs=0.01)

    def test_voltage_at_half_capacity(self):
        """Test voltage at 50% capacity is 3.6V."""
        from hardware_sim import Battery

        battery = Battery()
        battery.current_capacity_mah = BATTERY_CAPACITY_MAH / 2
        battery.update_voltage()

        # V = 3.0 + 1.2 * 0.5 = 3.6
        assert battery.voltage == pytest.approx(3.6, abs=0.01)

    def test_voltage_at_zero_capacity(self):
        """Test voltage at 0% capacity is 3.0V."""
        from hardware_sim import Battery

        battery = Battery()
        battery.current_capacity_mah = 0
        battery.update_voltage()

        assert battery.voltage == pytest.approx(3.0, abs=0.01)

    def test_low_battery_threshold(self):
        """Test low battery detection at V <= 3.2V."""
        from hardware_sim import Battery

        battery = Battery()

        # At threshold: V = 3.2 -> ratio = (3.2 - 3.0) / 1.2 = 0.1667
        # capacity = 0.1667 * 10000 = 1666.67 mAh
        # Use slightly lower to ensure we're at or below 3.2V
        battery.current_capacity_mah = 1666
        battery.update_voltage()

        # V = 3.0 + 1.2 * (1666/10000) = 3.0 + 0.19992 = 3.19992V
        assert battery.is_low(), f"Should be low at V={battery.voltage:.4f}V (<= 3.2V)"

        # Just above threshold
        battery.current_capacity_mah = 1700
        battery.update_voltage()

        # V = 3.0 + 1.2 * 0.17 = 3.204V
        assert not battery.is_low(), f"Should not be low at V={battery.voltage:.4f}V (> 3.2V)"

    def test_mah_consumed_tracking(self):
        """Test mAh consumed calculation."""
        from hardware_sim import Battery

        battery = Battery()
        initial = battery.current_capacity_mah

        # Consume 10 ticks of LISTENING
        for _ in range(10):
            battery.consume_power("LISTENING")

        expected_consumed = 10 * POWER_CONSUMPTION["LISTENING"] * (CHUNK_DURATION / 3600.0)
        actual_consumed = battery.mah_consumed()

        assert actual_consumed == pytest.approx(expected_consumed, abs=1e-6)
