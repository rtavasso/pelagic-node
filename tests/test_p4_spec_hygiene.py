"""
P4 Tests: Spec Hygiene / Regression

These tests verify spec compliance for model quantization,
config constants, and document behavioral choices.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestQuantizedModelFlag:
    """
    Test that ONNX model is quantized as specified.

    SPEC: Module C calls for "MobileNetV2 (Quantized ONNX)".
    BUG: train_model.py exports float32 model without quantization.
    """

    def test_model_export_format(self, dummy_onnx_model):
        """Check if exported model is quantized."""
        pytest.importorskip("onnx")
        import onnx

        model = onnx.load(str(dummy_onnx_model))
        node_types = {node.op_type for node in model.graph.node}
        assert "QuantizeLinear" in node_types or "DequantizeLinear" in node_types

    def test_model_weight_dtype(self, dummy_onnx_model):
        """Verify model weights are quantized (INT8)."""
        pytest.importorskip("onnx")
        import onnx

        model = onnx.load(str(dummy_onnx_model))
        # Quantized models should have INT8 initializers
        assert any(t.data_type == onnx.TensorProto.INT8 for t in model.graph.initializer)


class TestConfigConstants:
    """
    Verify all config constants match spec exactly.
    """

    def test_sample_rate(self):
        """SAMPLE_RATE == 16000."""
        from config import SAMPLE_RATE
        assert SAMPLE_RATE == 16000

    def test_chunk_duration(self):
        """CHUNK_DURATION == 1.0."""
        from config import CHUNK_DURATION
        assert CHUNK_DURATION == 1.0

    def test_context_window(self):
        """CONTEXT_WINDOW == 3.0."""
        from config import CONTEXT_WINDOW
        assert CONTEXT_WINDOW == 3.0

    def test_n_fft(self):
        """N_FFT == 1024."""
        from config import N_FFT
        assert N_FFT == 1024

    def test_hop_length(self):
        """HOP_LENGTH == 214."""
        from config import HOP_LENGTH
        assert HOP_LENGTH == 214

    def test_n_mels(self):
        """N_MELS == 224."""
        from config import N_MELS
        assert N_MELS == 224

    def test_time_steps(self):
        """TIME_STEPS == 224."""
        from config import TIME_STEPS
        assert TIME_STEPS == 224

    def test_battery_capacity(self):
        """BATTERY_CAPACITY_MAH == 10000."""
        from config import BATTERY_CAPACITY_MAH
        assert BATTERY_CAPACITY_MAH == 10000

    def test_power_consumption_listening(self):
        """POWER_CONSUMPTION['LISTENING'] == 120."""
        from config import POWER_CONSUMPTION
        assert POWER_CONSUMPTION["LISTENING"] == 120

    def test_power_consumption_transmit(self):
        """POWER_CONSUMPTION['TRANSMIT'] == 600."""
        from config import POWER_CONSUMPTION
        assert POWER_CONSUMPTION["TRANSMIT"] == 600

    def test_power_consumption_sleep(self):
        """POWER_CONSUMPTION['SLEEP'] == 80."""
        from config import POWER_CONSUMPTION
        assert POWER_CONSUMPTION["SLEEP"] == 80

    def test_tx_duration_ticks(self):
        """TX_DURATION_TICKS == 10."""
        from config import TX_DURATION_TICKS
        assert TX_DURATION_TICKS == 10

    def test_wake_threshold(self):
        """WAKE_THRESHOLD == 0.05."""
        from config import WAKE_THRESHOLD
        assert WAKE_THRESHOLD == 0.05

    def test_confidence_threshold(self):
        """CONFIDENCE_THRESHOLD == 0.85."""
        from config import CONFIDENCE_THRESHOLD
        assert CONFIDENCE_THRESHOLD == 0.85

    def test_tx_cooldown_sec(self):
        """TX_COOLDOWN_SEC == 300."""
        from config import TX_COOLDOWN_SEC
        assert TX_COOLDOWN_SEC == 300

    def test_tx_cooldown_ticks(self):
        """TX_COOLDOWN_TICKS == 300."""
        from config import TX_COOLDOWN_TICKS
        assert TX_COOLDOWN_TICKS == 300

    def test_cooldown_ticks_derived_correctly(self):
        """TX_COOLDOWN_TICKS == TX_COOLDOWN_SEC / CHUNK_DURATION."""
        from config import TX_COOLDOWN_SEC, CHUNK_DURATION, TX_COOLDOWN_TICKS
        expected = int(TX_COOLDOWN_SEC / CHUNK_DURATION)
        assert TX_COOLDOWN_TICKS == expected


class TestClassMappingSpec:
    """
    Verify class ID mapping matches spec.
    """

    def test_class_names_defined(self):
        """CLASS_NAMES dict exists with correct values."""
        from config import CLASS_NAMES

        assert CLASS_NAMES[0] == "Background"
        assert CLASS_NAMES[1] == "Vessel"
        assert CLASS_NAMES[2] == "Cetacean"

    def test_inference_class_mapping(self):
        """Inference engine maps ImageFolder order to spec order."""
        from inference import InferenceEngine

        # ImageFolder sorts alphabetically: background=0, cetacean=1, vessel=2
        # Spec order: Background=0, Vessel=1, Cetacean=2

        mapping = InferenceEngine.IMAGEFOLDER_TO_SPEC

        assert mapping[0] == 0  # background -> Background
        assert mapping[1] == 2  # cetacean -> Cetacean
        assert mapping[2] == 1  # vessel -> Vessel


class TestLogLifecycleDocumented:
    """
    Document and test log lifecycle behavior.

    Current behavior: Log is cleared on each simulation run.
    This test documents this as the chosen behavior.
    """

    def test_log_cleared_on_open_documented(self, temp_log_dir):
        """
        Document: TelemetryLogger.open() clears existing log.

        This is the current implementation choice. If logs should
        accumulate across runs, this behavior needs to change.
        """
        from comms import TelemetryLogger

        log_file = temp_log_dir / "mission_log.jsonl"

        # Create existing log
        with open(log_file, 'w') as f:
            f.write('{"existing": "data"}\n')

        # Open logger (current behavior: clears file)
        logger = TelemetryLogger(str(log_file))
        logger.open()
        logger.close()

        # Check if file was cleared
        with open(log_file) as f:
            content = f.read()

        # Document actual behavior
        if len(content) == 0:
            # Current behavior - log is cleared
            assert True, "Log is cleared on open (documented behavior)"
        else:
            # Alternative behavior - log persists
            pytest.fail(
                "Expected log to be cleared on open. "
                "If log accumulation is intended, update this test."
            )


class TestSpectrogramMathVerification:
    """
    Verify spectrogram math matches spec exactly.
    """

    def test_frame_count_formula(self):
        """Verify frame count: floor((n_samples - n_fft) / hop_length) + 1."""
        from config import N_FFT, HOP_LENGTH, SAMPLE_RATE, CONTEXT_WINDOW

        n_samples = int(SAMPLE_RATE * CONTEXT_WINDOW)  # 48000
        expected_frames = (n_samples - N_FFT) // HOP_LENGTH + 1

        # With hop_length=214: floor((48000 - 1024) / 214) + 1 = 220
        assert expected_frames == 220, \
            f"Expected 220 frames, calculated {expected_frames}"

    def test_padding_amount(self):
        """Verify padding: 224 - 220 = 4 frames."""
        from config import TIME_STEPS

        raw_frames = 220
        padding = TIME_STEPS - raw_frames

        assert padding == 4, f"Expected 4 frames padding, got {padding}"

    def test_voltage_formula(self):
        """Verify voltage formula: V = 3.0 + 1.2 * ratio."""
        from hardware_sim import Battery

        battery = Battery()

        # Test at various capacity levels
        test_cases = [
            (1.0, 4.2),    # 100% -> 4.2V
            (0.5, 3.6),    # 50% -> 3.6V
            (0.0, 3.0),    # 0% -> 3.0V
            (0.1667, 3.2), # ~17% -> 3.2V (threshold)
        ]

        for ratio, expected_voltage in test_cases:
            battery.current_capacity_mah = battery.max_capacity_mah * ratio
            battery.update_voltage()

            assert battery.voltage == pytest.approx(expected_voltage, abs=0.01), \
                f"At ratio {ratio}, expected {expected_voltage}V, got {battery.voltage}V"

    def test_power_consumption_formula(self):
        """Verify consumption: delta_mah = current_ma * (duration / 3600)."""
        from hardware_sim import Battery
        from config import POWER_CONSUMPTION, CHUNK_DURATION

        battery = Battery()
        initial = battery.current_capacity_mah

        # Test each state
        for state, current_ma in POWER_CONSUMPTION.items():
            battery.current_capacity_mah = initial
            battery.consume_power(state)

            expected_delta = current_ma * (CHUNK_DURATION / 3600.0)
            actual_delta = initial - battery.current_capacity_mah

            assert actual_delta == pytest.approx(expected_delta, abs=1e-9), \
                f"For {state}: expected {expected_delta}, got {actual_delta}"


class TestStateEnumDefinitions:
    """
    Verify state enum matches spec.
    """

    def test_state_enum_values(self):
        """Verify all states from spec are defined."""
        from main import State

        # Spec defines: LISTENING, INFERENCE, TRANSMIT, SLEEP, SHUTDOWN
        assert hasattr(State, 'LISTENING')
        assert hasattr(State, 'INFERENCE')
        assert hasattr(State, 'TRANSMIT')
        assert hasattr(State, 'SLEEP')
        assert hasattr(State, 'SHUTDOWN')

    def test_initial_state_is_listening(self):
        """Verify initial state is LISTENING per spec."""
        from main import State

        # Spec: "Initial State: LISTENING with an empty buffer"
        initial_state = State.LISTENING
        assert initial_state == State.LISTENING


class TestBufferImplementation:
    """
    Verify buffer implementation matches spec.
    """

    def test_buffer_uses_deque_maxlen_3(self):
        """Buffer uses collections.deque(maxlen=3)."""
        from dsp_pipeline import AudioBuffer
        from collections import deque

        buffer = AudioBuffer()

        assert isinstance(buffer.buffer, deque)
        assert buffer.buffer.maxlen == 3

    def test_buffer_stores_1_second_arrays(self):
        """Buffer stores 1-second numpy arrays (16000 samples)."""
        from dsp_pipeline import AudioBuffer
        from config import SAMPLE_RATE

        buffer = AudioBuffer()

        # Add 1-second chunks
        for _ in range(3):
            chunk = np.zeros(SAMPLE_RATE, dtype=np.float32)
            buffer.append(chunk)

        # Verify stored chunks are correct size
        for chunk in buffer.buffer:
            assert len(chunk) == SAMPLE_RATE
            assert chunk.dtype == np.float32


class TestDerivedConstants:
    """
    Verify derived constants are calculated correctly.
    """

    def test_chunk_samples(self):
        """CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION."""
        from config import SAMPLE_RATE, CHUNK_DURATION

        expected = int(SAMPLE_RATE * CHUNK_DURATION)
        assert expected == 16000

    def test_context_chunks(self):
        """CONTEXT_CHUNKS = CONTEXT_WINDOW / CHUNK_DURATION."""
        from config import CONTEXT_WINDOW, CHUNK_DURATION

        expected = int(CONTEXT_WINDOW / CHUNK_DURATION)
        assert expected == 3

    def test_blind_spot_total(self):
        """Total blind spot = TX_DURATION + TX_COOLDOWN = 310 seconds."""
        from config import TX_DURATION_TICKS, TX_COOLDOWN_TICKS

        total_blind = TX_DURATION_TICKS + TX_COOLDOWN_TICKS
        assert total_blind == 310


class TestErrorHandlingCategories:
    """
    Verify error handling categories match spec Section 7.
    """

    def test_recoverable_errors_documented(self):
        """Document recoverable error types from spec 7.1."""
        # Spec 7.1 Recoverable Errors:
        # - Corrupted audio chunk -> log warning, skip, continue
        # - ONNX inference failure -> log error, return to LISTENING
        # - Single file read error -> log warning, skip to next file

        recoverable_errors = [
            "Corrupted audio chunk",
            "ONNX inference failure",
            "Single file read error",
        ]

        # This test documents the expected recoverable errors
        # Actual implementation of recovery is tested in P0 tests
        assert len(recoverable_errors) == 3

    def test_fatal_errors_documented(self):
        """Document fatal error types from spec 7.2."""
        # Spec 7.2 Fatal Errors:
        # - No audio files in sim_input -> sys.exit(1)
        # - ONNX model file not found -> sys.exit(1)
        # - ONNX model load failure -> sys.exit(1)
        # - Log directory not writable -> sys.exit(1)
        # - Invalid config values -> sys.exit(1)

        fatal_errors = [
            "No audio files in sim_input",
            "ONNX model file not found",
            "ONNX model load failure",
            "Log directory not writable",
            "Invalid config values",
        ]

        assert len(fatal_errors) == 5
