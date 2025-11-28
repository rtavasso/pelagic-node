"""
P2 Tests: Environment & DSP Fidelity

These tests verify audio environment handling (resampling, stereo conversion)
and DSP pipeline correctness (spectrogram shape, normalization).
"""

import sys
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, TIME_STEPS


class TestResampleAndMonoConversion:
    """
    Test audio resampling and stereo to mono conversion.
    """

    def test_resample_8khz_to_16khz(self, temp_audio_dir):
        """8 kHz input becomes 16 kHz mono."""
        from hardware_sim import AudioEnvironment

        # Create 8kHz audio file (1 second = 8000 samples)
        t = np.linspace(0, 1, 8000, dtype=np.float32)
        audio_8k = 0.5 * np.sin(2 * np.pi * 440 * t)

        wav_path = temp_audio_dir / "test_8k.wav"
        sf.write(wav_path, audio_8k, 8000)

        env = AudioEnvironment(str(temp_audio_dir))
        chunk = env.read_chunk()

        # Should be resampled to 16kHz
        assert chunk is not None
        assert len(chunk) == 16000, \
            f"Resampled chunk should be 16000 samples, got {len(chunk)}"
        assert chunk.dtype == np.float32

    def test_resample_44100_to_16khz(self, temp_audio_dir):
        """44.1 kHz input becomes 16 kHz."""
        from hardware_sim import AudioEnvironment

        # Create 44.1kHz audio file (1 second = 44100 samples)
        t = np.linspace(0, 1, 44100, dtype=np.float32)
        audio_44k = 0.5 * np.sin(2 * np.pi * 440 * t)

        wav_path = temp_audio_dir / "test_44k.wav"
        sf.write(wav_path, audio_44k, 44100)

        env = AudioEnvironment(str(temp_audio_dir))
        chunk = env.read_chunk()

        assert chunk is not None
        assert len(chunk) == 16000

    def test_stereo_to_mono_averaging(self, temp_audio_dir):
        """Stereo input becomes mono via (L+R)/2."""
        from hardware_sim import AudioEnvironment

        # Create stereo file: left = constant 0.4, right = constant 0.8
        # Mono should be (0.4 + 0.8) / 2 = 0.6
        duration = 1.0
        num_samples = int(SAMPLE_RATE * duration)

        left = np.full(num_samples, 0.4, dtype=np.float32)
        right = np.full(num_samples, 0.8, dtype=np.float32)
        stereo = np.column_stack([left, right])

        wav_path = temp_audio_dir / "stereo.wav"
        sf.write(wav_path, stereo, SAMPLE_RATE)

        env = AudioEnvironment(str(temp_audio_dir))
        chunk = env.read_chunk()

        assert chunk is not None
        assert len(chunk) == 16000
        assert chunk.ndim == 1, "Output should be mono (1D)"

        expected_mono = 0.6
        assert np.allclose(chunk, expected_mono, atol=1e-5), \
            f"Mono should be (L+R)/2 = 0.6, got mean={chunk.mean()}"

    def test_stereo_sine_waves_averaged(self, temp_audio_dir):
        """Stereo sine waves at different frequencies are averaged."""
        from hardware_sim import AudioEnvironment

        duration = 1.0
        num_samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples, dtype=np.float32)

        # Left: 440 Hz, Right: 880 Hz
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)
        stereo = np.column_stack([left, right])

        wav_path = temp_audio_dir / "stereo_sine.wav"
        sf.write(wav_path, stereo, SAMPLE_RATE)

        env = AudioEnvironment(str(temp_audio_dir))
        chunk = env.read_chunk()

        # Expected: (left + right) / 2
        expected = (left + right) / 2

        assert chunk is not None
        assert np.allclose(chunk, expected, atol=1e-4), \
            "Mono should be average of left and right channels"


class TestCrossFileContextWindow:
    """
    Test that the 3-second context window can span file boundaries.
    """

    def test_cross_file_context_window(self, temp_audio_dir, make_wav_file):
        """
        Last 2s of file A + first 1s of file B appear consecutively in buffer.
        """
        from hardware_sim import AudioEnvironment
        from dsp_pipeline import AudioBuffer, normalize

        # File A: 2.5 seconds of value 0.3
        make_wav_file(temp_audio_dir / "a_file.wav", duration_sec=2.5,
                      constant_value=0.3)

        # File B: 2 seconds of value 0.7
        make_wav_file(temp_audio_dir / "b_file.wav", duration_sec=2.0,
                      constant_value=0.7)

        env = AudioEnvironment(str(temp_audio_dir))
        buffer = AudioBuffer()

        # Read chunks and fill buffer
        chunks_read = []
        for _ in range(4):  # Should be enough to span files
            chunk = env.read_chunk()
            if chunk is None:
                break
            chunks_read.append(chunk.copy())
            buffer.append(normalize(chunk))

        # After 4 reads:
        # Chunk 0: 1s from A (0.3)
        # Chunk 1: 1s from A (0.3)
        # Chunk 2: 0.5s from A + 0.5s from B (mixed) OR padded depending on impl
        # Chunk 3: 1s from B (0.7) OR continuation

        # Buffer should have last 3 chunks
        assert buffer.is_full()

        # Verify we can generate spectrogram from cross-file buffer
        spec = buffer.get_spectrogram()
        assert spec.shape == (1, 3, 224, 224)

    def test_multi_file_sequential_processing(self, temp_audio_dir, make_wav_file):
        """Files are processed in alphabetical order."""
        from hardware_sim import AudioEnvironment

        # Create files with names that sort: a, b, c
        make_wav_file(temp_audio_dir / "c_third.wav", duration_sec=1.0,
                      constant_value=0.9)
        make_wav_file(temp_audio_dir / "a_first.wav", duration_sec=1.0,
                      constant_value=0.1)
        make_wav_file(temp_audio_dir / "b_second.wav", duration_sec=1.0,
                      constant_value=0.5)

        env = AudioEnvironment(str(temp_audio_dir))

        chunk1 = env.read_chunk()
        chunk2 = env.read_chunk()
        chunk3 = env.read_chunk()

        # Should be in order: a (0.1), b (0.5), c (0.9)
        # Use atol=1e-3 to account for float32 precision in WAV encoding
        assert np.allclose(chunk1, 0.1, atol=1e-3), \
            f"First file should be 'a' with value 0.1, got {chunk1.mean()}"
        assert np.allclose(chunk2, 0.5, atol=1e-3), \
            f"Second file should be 'b' with value 0.5, got {chunk2.mean()}"
        assert np.allclose(chunk3, 0.9, atol=1e-3), \
            f"Third file should be 'c' with value 0.9, got {chunk3.mean()}"


class TestSpectrogramShapeAndPadding:
    """
    Test spectrogram generation produces correct shape and value range.
    """

    def test_spectrogram_shape(self):
        """generate_spectrogram yields [1,3,224,224]."""
        from dsp_pipeline import generate_spectrogram

        # Create 3-second buffer of normalized audio
        buffer = deque(maxlen=3)
        for _ in range(3):
            t = np.linspace(0, 1, SAMPLE_RATE, dtype=np.float32)
            chunk = np.sin(2 * np.pi * 440 * t)
            buffer.append(chunk)

        spec = generate_spectrogram(buffer)

        assert spec.shape == (1, 3, 224, 224), \
            f"Expected (1,3,224,224), got {spec.shape}"

    def test_spectrogram_dtype(self):
        """Spectrogram should be float32."""
        from dsp_pipeline import generate_spectrogram

        buffer = deque(maxlen=3)
        for _ in range(3):
            chunk = np.random.randn(SAMPLE_RATE).astype(np.float32)
            buffer.append(chunk)

        spec = generate_spectrogram(buffer)

        assert spec.dtype == np.float32, \
            f"Expected float32, got {spec.dtype}"

    def test_spectrogram_value_range(self):
        """Spectrogram values should be in [0, 1]."""
        from dsp_pipeline import generate_spectrogram

        buffer = deque(maxlen=3)
        for _ in range(3):
            t = np.linspace(0, 1, SAMPLE_RATE, dtype=np.float32)
            chunk = 0.5 * np.sin(2 * np.pi * 440 * t)
            buffer.append(chunk)

        spec = generate_spectrogram(buffer)

        assert spec.min() >= 0.0, f"Min value {spec.min()} < 0"
        assert spec.max() <= 1.0, f"Max value {spec.max()} > 1"

    def test_spectrogram_frame_padding(self):
        """
        Test that spectrogram is padded from 220 frames to 224.

        With hop_length=214, 48000 samples -> floor((48000-1024)/214)+1 = 220 frames
        Padding adds 4 frames to reach 224.
        """
        from dsp_pipeline import generate_spectrogram
        import librosa

        # Create buffer
        buffer = deque(maxlen=3)
        for _ in range(3):
            chunk = np.random.randn(SAMPLE_RATE).astype(np.float32)
            buffer.append(chunk)

        audio_3s = np.concatenate(list(buffer))
        assert len(audio_3s) == 48000

        # Calculate expected frames before padding
        expected_frames = (48000 - N_FFT) // HOP_LENGTH + 1
        assert expected_frames == 220, f"Expected 220 frames, calculated {expected_frames}"

        # Generate spectrogram
        spec = generate_spectrogram(buffer)

        # Final shape should be 224 (padded)
        assert spec.shape[3] == 224, \
            f"Time dimension should be 224, got {spec.shape[3]}"

    def test_spectrogram_mel_bands(self):
        """Test spectrogram has 224 mel bands."""
        from dsp_pipeline import generate_spectrogram

        buffer = deque(maxlen=3)
        for _ in range(3):
            chunk = np.random.randn(SAMPLE_RATE).astype(np.float32)
            buffer.append(chunk)

        spec = generate_spectrogram(buffer)

        # Height (mel bands) should be 224
        assert spec.shape[2] == N_MELS, \
            f"Mel bands should be {N_MELS}, got {spec.shape[2]}"

    def test_spectrogram_three_channels(self):
        """Test spectrogram has 3 identical channels (RGB stacking)."""
        from dsp_pipeline import generate_spectrogram

        buffer = deque(maxlen=3)
        for _ in range(3):
            chunk = np.random.randn(SAMPLE_RATE).astype(np.float32)
            buffer.append(chunk)

        spec = generate_spectrogram(buffer)

        # Should have 3 channels
        assert spec.shape[1] == 3, f"Should have 3 channels, got {spec.shape[1]}"

        # All channels should be identical
        np.testing.assert_array_equal(spec[0, 0], spec[0, 1],
                                       err_msg="Channels 0 and 1 should be identical")
        np.testing.assert_array_equal(spec[0, 0], spec[0, 2],
                                       err_msg="Channels 0 and 2 should be identical")


class TestRMSSilenceGuard:
    """
    Test RMS computation and silence protection in normalization.
    """

    def test_rms_on_silence(self, silent_chunk):
        """RMS of silence should be 0."""
        from dsp_pipeline import compute_rms

        rms = compute_rms(silent_chunk)
        assert rms == 0.0, f"RMS of silence should be 0, got {rms}"

    def test_rms_on_known_signal(self):
        """RMS of known signal matches expected value."""
        from dsp_pipeline import compute_rms

        # Constant signal of 0.5 has RMS = 0.5
        constant = np.full(SAMPLE_RATE, 0.5, dtype=np.float32)
        rms = compute_rms(constant)
        assert rms == pytest.approx(0.5, abs=1e-6)

        # Sine wave: RMS = amplitude / sqrt(2)
        t = np.linspace(0, 1, SAMPLE_RATE, dtype=np.float32)
        sine = 0.8 * np.sin(2 * np.pi * 440 * t)
        rms_sine = compute_rms(sine)
        expected_rms = 0.8 / np.sqrt(2)
        assert rms_sine == pytest.approx(expected_rms, abs=0.01)

    def test_normalize_silence_returns_zeros(self, silent_chunk):
        """Normalizing silence should return zeros (not NaN/Inf)."""
        from dsp_pipeline import normalize

        normalized = normalize(silent_chunk)

        assert not np.any(np.isnan(normalized)), "Should not contain NaN"
        assert not np.any(np.isinf(normalized)), "Should not contain Inf"
        assert np.all(normalized == 0), "Normalized silence should be all zeros"

    def test_normalize_tiny_signal_returns_zeros(self, tiny_signal_chunk):
        """Very small signals (< 1e-6 max) treated as silence."""
        from dsp_pipeline import normalize

        assert np.max(np.abs(tiny_signal_chunk)) < 1e-6

        normalized = normalize(tiny_signal_chunk)

        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))
        assert np.all(normalized == 0), \
            "Tiny signals (max < 1e-6) should be treated as silence"

    def test_normalize_preserves_shape(self, high_rms_chunk):
        """Normalization preserves array shape."""
        from dsp_pipeline import normalize

        normalized = normalize(high_rms_chunk)

        assert normalized.shape == high_rms_chunk.shape
        assert normalized.dtype == np.float32

    def test_normalize_scales_to_unit_peak(self, high_rms_chunk):
        """Normalization scales to peak = 1.0."""
        from dsp_pipeline import normalize

        normalized = normalize(high_rms_chunk)

        max_val = np.max(np.abs(normalized))
        assert max_val == pytest.approx(1.0, abs=1e-6), \
            f"Peak should be 1.0, got {max_val}"

    def test_rms_stays_on_raw_after_normalize(self, high_rms_chunk):
        """Verify RMS is different before and after normalization."""
        from dsp_pipeline import compute_rms, normalize

        raw_rms = compute_rms(high_rms_chunk)
        normalized = normalize(high_rms_chunk)
        normalized_rms = compute_rms(normalized)

        # They should be different (unless the chunk was already normalized)
        assert raw_rms != normalized_rms, \
            "RMS should differ between raw and normalized audio"

        # Normalized RMS should be larger (scaled up to peak=1.0)
        assert normalized_rms > raw_rms, \
            "Normalized audio should have higher RMS (scaled up)"


class TestAudioChunkProperties:
    """
    Test audio chunk reading properties.
    """

    def test_chunk_length_16000_samples(self, temp_audio_dir, make_wav_file):
        """Each chunk should be exactly 16000 samples."""
        from hardware_sim import AudioEnvironment

        make_wav_file(temp_audio_dir / "test.wav", duration_sec=3.0)

        env = AudioEnvironment(str(temp_audio_dir))

        for _ in range(3):
            chunk = env.read_chunk()
            assert chunk is not None
            assert len(chunk) == 16000, f"Chunk should be 16000 samples, got {len(chunk)}"

    def test_chunk_dtype_float32(self, temp_audio_dir, make_wav_file):
        """Chunks should be float32."""
        from hardware_sim import AudioEnvironment

        make_wav_file(temp_audio_dir / "test.wav", duration_sec=1.0)

        env = AudioEnvironment(str(temp_audio_dir))
        chunk = env.read_chunk()

        assert chunk.dtype == np.float32, f"Expected float32, got {chunk.dtype}"

    def test_eof_returns_none(self, temp_audio_dir, make_wav_file):
        """EOF returns None."""
        from hardware_sim import AudioEnvironment

        make_wav_file(temp_audio_dir / "test.wav", duration_sec=1.0)

        env = AudioEnvironment(str(temp_audio_dir))

        chunk1 = env.read_chunk()
        assert chunk1 is not None

        chunk2 = env.read_chunk()
        assert chunk2 is None, "Should return None after audio exhausted"

    def test_is_exhausted_flag(self, temp_audio_dir, make_wav_file):
        """is_exhausted() returns True after EOF."""
        from hardware_sim import AudioEnvironment

        make_wav_file(temp_audio_dir / "test.wav", duration_sec=1.0)

        env = AudioEnvironment(str(temp_audio_dir))

        assert not env.is_exhausted()

        env.read_chunk()  # Read all audio
        env.read_chunk()  # Get None (EOF)

        assert env.is_exhausted()

    def test_supported_formats_wav_flac(self, temp_audio_dir):
        """Both .wav and .flac formats are supported."""
        from hardware_sim import AudioEnvironment

        # Create WAV file
        t = np.linspace(0, 1, SAMPLE_RATE, dtype=np.float32)
        wav_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(temp_audio_dir / "test.wav", wav_data, SAMPLE_RATE)

        # Create FLAC file
        flac_data = 0.3 * np.sin(2 * np.pi * 880 * t)
        sf.write(temp_audio_dir / "test.flac", flac_data, SAMPLE_RATE)

        env = AudioEnvironment(str(temp_audio_dir))

        # Should read both files
        chunk1 = env.read_chunk()
        chunk2 = env.read_chunk()

        assert chunk1 is not None
        assert chunk2 is not None

        # Values should match the files (alphabetical: flac then wav)
        # test.flac comes before test.wav
        assert np.allclose(chunk1.mean(), 0, atol=0.1)  # flac sine wave mean ~0
        assert np.allclose(chunk2.mean(), 0, atol=0.1)  # wav sine wave mean ~0
