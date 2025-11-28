"""
Pytest fixtures for pelagic-node test suite.

Provides:
- Temporary directories for audio/logs/models
- Stub inference engine for controlled testing
- Deterministic WAV file generators
- Battery and environment mocks
"""

import sys
import os
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import deque

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import SAMPLE_RATE, CHUNK_DURATION


# =============================================================================
# Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_audio_dir(tmp_path):
    """Create a temporary directory for audio files."""
    audio_dir = tmp_path / "sim_input"
    audio_dir.mkdir()
    return audio_dir


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary directory for log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


# =============================================================================
# Audio Generation Fixtures
# =============================================================================

@pytest.fixture
def make_wav_file():
    """Factory fixture to create WAV files with specific characteristics."""
    def _make_wav(path, duration_sec, sample_rate=SAMPLE_RATE,
                  amplitude=0.1, frequency=440, stereo=False, constant_value=None):
        """
        Create a WAV file.

        Args:
            path: Output file path
            duration_sec: Duration in seconds
            sample_rate: Sample rate (default 16kHz)
            amplitude: Peak amplitude (0-1)
            frequency: Tone frequency in Hz (ignored if constant_value set)
            stereo: If True, create stereo file
            constant_value: If set, fill with this constant instead of sine
        """
        num_samples = int(sample_rate * duration_sec)

        if constant_value is not None:
            data = np.full(num_samples, constant_value, dtype=np.float32)
        else:
            t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)
            data = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

        if stereo:
            # Left channel: original, Right channel: different frequency
            if constant_value is not None:
                right = np.full(num_samples, constant_value * 0.5, dtype=np.float32)
            else:
                t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)
                right = (amplitude * np.sin(2 * np.pi * frequency * 2 * t)).astype(np.float32)
            data = np.column_stack([data, right])

        sf.write(path, data, sample_rate)
        return path

    return _make_wav


@pytest.fixture
def make_high_rms_wav(make_wav_file):
    """Create a WAV file with RMS above WAKE_THRESHOLD (0.05)."""
    def _make(path, duration_sec=1.0):
        # Amplitude 0.15 gives RMS ~0.106 for sine wave (RMS = amp / sqrt(2))
        return make_wav_file(path, duration_sec, amplitude=0.15, frequency=440)
    return _make


@pytest.fixture
def make_low_rms_wav(make_wav_file):
    """Create a WAV file with RMS below WAKE_THRESHOLD (0.05)."""
    def _make(path, duration_sec=1.0):
        # Amplitude 0.03 gives RMS ~0.021 for sine wave
        return make_wav_file(path, duration_sec, amplitude=0.03, frequency=440)
    return _make


@pytest.fixture
def make_silent_wav(make_wav_file):
    """Create a silent WAV file (all zeros)."""
    def _make(path, duration_sec=1.0):
        return make_wav_file(path, duration_sec, constant_value=0.0)
    return _make


# =============================================================================
# Stub/Mock Fixtures
# =============================================================================

@pytest.fixture
def stub_inference_engine():
    """
    Create a stub inference engine that returns controlled outputs.

    Usage:
        engine = stub_inference_engine()
        engine.set_response(class_id=1, confidence=0.95)
        result = engine.run(spectrogram)
    """
    class StubInferenceEngine:
        def __init__(self):
            self.call_count = 0
            self.last_input = None
            self._class_id = 0
            self._confidence = 0.5
            self._should_raise = False
            self._raise_exception = None

        def set_response(self, class_id=0, confidence=0.5):
            """Set the response for next run() call."""
            self._class_id = class_id
            self._confidence = confidence

        def set_raise(self, exception):
            """Make run() raise an exception."""
            self._should_raise = True
            self._raise_exception = exception

        def run(self, spectrogram):
            """Stub inference - returns configured response."""
            self.call_count += 1
            self.last_input = spectrogram

            if self._should_raise:
                raise self._raise_exception

            return self._class_id, self._confidence

        def reset(self):
            """Reset call count and last input."""
            self.call_count = 0
            self.last_input = None

    return StubInferenceEngine()


@pytest.fixture
def stub_audio_environment():
    """
    Create a stub audio environment that yields controlled chunks.

    Usage:
        env = stub_audio_environment()
        env.set_chunks([chunk1, chunk2, None])  # None signals EOF
    """
    class StubAudioEnvironment:
        def __init__(self):
            self._chunks = []
            self._index = 0
            self.read_count = 0

        def set_chunks(self, chunks):
            """Set the sequence of chunks to return."""
            self._chunks = chunks
            self._index = 0

        def add_chunk(self, chunk):
            """Add a chunk to the queue."""
            self._chunks.append(chunk)

        def read_chunk(self):
            """Return next chunk or None if exhausted."""
            self.read_count += 1
            if self._index >= len(self._chunks):
                return None
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk

        def is_exhausted(self):
            return self._index >= len(self._chunks) or \
                   (self._index > 0 and self._chunks[self._index - 1] is None)

        def reset(self):
            self._index = 0
            self.read_count = 0

    return StubAudioEnvironment()


@pytest.fixture
def mock_battery():
    """Create a mock battery for testing power consumption."""
    class MockBattery:
        def __init__(self):
            self.consume_calls = []
            self.update_voltage_calls = 0
            self._voltage = 4.2
            self.current_capacity_mah = 10000
            self.max_capacity_mah = 10000

        def consume_power(self, state):
            self.consume_calls.append(state)

        def update_voltage(self):
            self.update_voltage_calls += 1

        @property
        def voltage(self):
            return self._voltage

        def set_voltage(self, v):
            self._voltage = v

        def is_low(self):
            return self._voltage <= 3.2

        @property
        def capacity_percent(self):
            return 100.0 * self.current_capacity_mah / self.max_capacity_mah

        def mah_consumed(self):
            return self.max_capacity_mah - self.current_capacity_mah

    return MockBattery()


# =============================================================================
# Audio Chunk Fixtures
# =============================================================================

@pytest.fixture
def high_rms_chunk():
    """Generate a 1-second chunk with RMS > 0.05."""
    # Sine wave with amplitude 0.15 -> RMS ~0.106
    t = np.linspace(0, 1.0, SAMPLE_RATE, dtype=np.float32)
    return 0.15 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def low_rms_chunk():
    """Generate a 1-second chunk with RMS < 0.05."""
    # Sine wave with amplitude 0.03 -> RMS ~0.021
    t = np.linspace(0, 1.0, SAMPLE_RATE, dtype=np.float32)
    return 0.03 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def silent_chunk():
    """Generate a 1-second silent chunk."""
    return np.zeros(SAMPLE_RATE, dtype=np.float32)


@pytest.fixture
def tiny_signal_chunk():
    """Generate a 1-second chunk with very small values (< 1e-6 max)."""
    return np.full(SAMPLE_RATE, 1e-7, dtype=np.float32)


# =============================================================================
# Dummy ONNX Model Fixture
# =============================================================================

@pytest.fixture
def dummy_onnx_model(temp_model_dir):
    """
    Create a minimal quantized-style ONNX model for testing.

    Includes QuantizeLinear/DequantizeLinear nodes and an INT8 initializer
    to satisfy quantization checks.
    """
    try:
        import onnx
        from onnx import helper, TensorProto

        # Input/Output
        X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])

        # Initializers for quantization
        scale_init = helper.make_tensor('scale', TensorProto.FLOAT, [1], [1.0])
        zp_init = helper.make_tensor('zero_point', TensorProto.UINT8, [1], [0])
        int8_weight = helper.make_tensor('int8_weight', TensorProto.INT8, [1], [1])

        q_node = helper.make_node(
            'QuantizeLinear',
            inputs=['input', 'scale', 'zero_point'],
            outputs=['quant']
        )

        dq_node = helper.make_node(
            'DequantizeLinear',
            inputs=['quant', 'scale', 'zero_point'],
            outputs=['dequant']
        )

        gap_node = helper.make_node('GlobalAveragePool', inputs=['dequant'], outputs=['pooled'])
        flatten_node = helper.make_node('Flatten', inputs=['pooled'], outputs=['flat'], axis=1)
        identity_node = helper.make_node('Identity', inputs=['flat'], outputs=['output'])

        graph = helper.make_graph(
            [q_node, dq_node, gap_node, flatten_node, identity_node],
            'quant_stub_model',
            [X],
            [Y],
            initializer=[scale_init, zp_init, int8_weight],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
        model_path = temp_model_dir / "classifier.onnx"
        onnx.save(model, str(model_path))

        return model_path

    except ImportError:
        pytest.skip("onnx package required for this test")


# =============================================================================
# Integration Test Helpers
# =============================================================================

@pytest.fixture
def simulation_harness(tmp_path, stub_inference_engine):
    """
    Set up a complete simulation environment for integration tests.

    Returns a dict with paths and stubs configured.
    """
    # Create directory structure
    sim_input = tmp_path / "data" / "sim_input"
    sim_input.mkdir(parents=True)

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    return {
        'tmp_path': tmp_path,
        'sim_input': sim_input,
        'logs_dir': logs_dir,
        'models_dir': models_dir,
        'log_file': logs_dir / "mission_log.jsonl",
        'engine': stub_inference_engine
    }
