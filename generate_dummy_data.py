#!/usr/bin/env python3
"""
Synthetic data generator for marine acoustic classifier training.

Generates three classes of audio:
- Background: Low-amplitude pink noise
- Vessel: Low-frequency engine tones (50-200 Hz) with harmonics
- Cetacean: High-frequency chirps/clicks (2-8 kHz)

Outputs:
- .wav files to data/raw_synthetic/
- .png spectrograms to data/processed_spectrograms/train/ and val/
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, CONTEXT_WINDOW

# Training dependencies
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image


# --- CONFIGURATION ---
NUM_SAMPLES_PER_CLASS = 100
TRAIN_SPLIT = 0.8
AUDIO_DURATION = CONTEXT_WINDOW  # 3 seconds per sample
NUM_AUDIO_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)  # 48000

# Output directories
RAW_DIR = Path("./data/raw_synthetic")
TRAIN_DIR = Path("./data/processed_spectrograms/train")
VAL_DIR = Path("./data/processed_spectrograms/val")


def generate_background(duration_samples: int) -> np.ndarray:
    """Generate low-amplitude pink noise (background ocean sounds)."""
    # Generate white noise
    white = np.random.randn(duration_samples)

    # Apply pink noise filter (1/f spectrum)
    # Simple approximation using cumulative sum and high-pass
    pink = np.cumsum(white)
    # Remove DC and low-frequency drift
    pink = pink - np.convolve(pink, np.ones(1000)/1000, mode='same')

    # Normalize and scale to low amplitude (RMS ~ 0.02-0.04)
    pink = pink / (np.max(np.abs(pink)) + 1e-6)
    amplitude = np.random.uniform(0.02, 0.04)
    pink = pink * amplitude

    return pink.astype(np.float32)


def generate_vessel(duration_samples: int) -> np.ndarray:
    """Generate vessel engine sounds (low-frequency tones with harmonics)."""
    t = np.linspace(0, duration_samples / SAMPLE_RATE, duration_samples)

    # Fundamental frequency (50-150 Hz typical for large vessels)
    fund_freq = np.random.uniform(50, 150)

    # Generate fundamental + harmonics
    signal = np.zeros(duration_samples)
    for harmonic in range(1, 6):
        freq = fund_freq * harmonic
        amp = 1.0 / harmonic  # Decreasing amplitude for higher harmonics
        phase = np.random.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * freq * t + phase)

    # Add some broadband noise (cavitation, turbulence)
    noise = np.random.randn(duration_samples) * 0.1
    signal += noise

    # Normalize to moderate amplitude (RMS ~ 0.1-0.3)
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    amplitude = np.random.uniform(0.1, 0.3)
    signal = signal * amplitude

    return signal.astype(np.float32)


def generate_cetacean(duration_samples: int) -> np.ndarray:
    """Generate cetacean sounds (high-frequency clicks and whistles)."""
    t = np.linspace(0, duration_samples / SAMPLE_RATE, duration_samples)
    signal = np.zeros(duration_samples)

    # Add several click trains
    num_click_trains = np.random.randint(2, 5)
    for _ in range(num_click_trains):
        # Click train parameters
        start_time = np.random.uniform(0, 2.0)
        click_rate = np.random.uniform(5, 20)  # Clicks per second
        num_clicks = int(np.random.uniform(5, 15))

        for i in range(num_clicks):
            click_time = start_time + i / click_rate
            if click_time >= AUDIO_DURATION:
                break

            # Click is a short high-frequency burst
            click_center = int(click_time * SAMPLE_RATE)
            click_duration = int(0.002 * SAMPLE_RATE)  # 2ms click

            if click_center + click_duration < duration_samples:
                click_freq = np.random.uniform(3000, 8000)
                click_t = np.arange(click_duration) / SAMPLE_RATE
                # Gaussian-windowed sinusoid
                window = np.exp(-((click_t - 0.001) ** 2) / (2 * 0.0003 ** 2))
                click = window * np.sin(2 * np.pi * click_freq * click_t)
                signal[click_center:click_center + click_duration] += click

    # Add frequency-modulated whistle
    if np.random.random() > 0.3:
        whistle_start = np.random.uniform(0.5, 2.0)
        whistle_duration = np.random.uniform(0.3, 0.8)
        whistle_samples = int(whistle_duration * SAMPLE_RATE)
        start_idx = int(whistle_start * SAMPLE_RATE)

        if start_idx + whistle_samples < duration_samples:
            whistle_t = np.arange(whistle_samples) / SAMPLE_RATE
            # Frequency sweep (upsweep or downsweep)
            f_start = np.random.uniform(2000, 5000)
            f_end = np.random.uniform(4000, 8000)
            freq = f_start + (f_end - f_start) * whistle_t / whistle_duration
            phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
            # Amplitude envelope
            envelope = np.sin(np.pi * whistle_t / whistle_duration)
            whistle = envelope * np.sin(phase) * 0.3
            signal[start_idx:start_idx + whistle_samples] += whistle

    # Add light background noise
    signal += np.random.randn(duration_samples) * 0.02

    # Normalize to moderate amplitude (RMS ~ 0.08-0.2)
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    amplitude = np.random.uniform(0.08, 0.2)
    signal = signal * amplitude

    return signal.astype(np.float32)


def audio_to_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to mel-spectrogram matching the inference pipeline.

    Returns: np.ndarray of shape [224, 224] normalized to [0, 1]
    """
    # Generate mel-spectrogram with exact parameters from spec
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        window='hann',
        center=False,
        power=2.0,
        fmin=20.0,
        fmax=8000.0
    )

    # Convert to log scale (dB)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalize to [0, 1]
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)

    # Pad to 224 time steps (from ~220)
    current_frames = S_norm.shape[1]
    if current_frames < 224:
        pad_width = 224 - current_frames
        S_norm = np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant')
    elif current_frames > 224:
        S_norm = S_norm[:, :224]

    return S_norm


def save_spectrogram_image(spectrogram: np.ndarray, path: Path):
    """Save spectrogram as PNG image."""
    # Convert to 8-bit grayscale
    img_data = (spectrogram * 255).astype(np.uint8)

    # Flip vertically (low frequencies at bottom)
    img_data = np.flipud(img_data)

    # Save as grayscale PNG
    img = Image.fromarray(img_data, mode='L')
    # Convert to RGB (3 channels) for model compatibility
    img_rgb = img.convert('RGB')
    img_rgb.save(path)


def main():
    """Generate synthetic dataset."""
    print("=" * 60)
    print("Synthetic Data Generator for Marine Acoustic Classifier")
    print("=" * 60)

    # Create directories
    for d in [RAW_DIR, TRAIN_DIR / "background", TRAIN_DIR / "vessel",
              TRAIN_DIR / "cetacean", VAL_DIR / "background",
              VAL_DIR / "vessel", VAL_DIR / "cetacean"]:
        d.mkdir(parents=True, exist_ok=True)

    # Class generators
    generators = {
        "background": generate_background,
        "vessel": generate_vessel,
        "cetacean": generate_cetacean
    }

    np.random.seed(42)  # Reproducibility

    for class_name, generator in generators.items():
        print(f"\nGenerating {NUM_SAMPLES_PER_CLASS} samples for class: {class_name}")

        train_count = int(NUM_SAMPLES_PER_CLASS * TRAIN_SPLIT)

        for i in range(NUM_SAMPLES_PER_CLASS):
            # Generate audio
            audio = generator(NUM_AUDIO_SAMPLES)

            # Save raw audio
            wav_path = RAW_DIR / f"{class_name}_{i:04d}.wav"
            sf.write(wav_path, audio, SAMPLE_RATE)

            # Generate and save spectrogram
            spectrogram = audio_to_spectrogram(audio)

            # Determine train/val split
            if i < train_count:
                out_dir = TRAIN_DIR / class_name
            else:
                out_dir = VAL_DIR / class_name

            png_path = out_dir / f"{class_name}_{i:04d}.png"
            save_spectrogram_image(spectrogram, png_path)

            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{NUM_SAMPLES_PER_CLASS}")

    # Print summary
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Raw audio files: {RAW_DIR}")
    print(f"Training spectrograms: {TRAIN_DIR}")
    print(f"Validation spectrograms: {VAL_DIR}")

    # Count files
    for split_dir, split_name in [(TRAIN_DIR, "Train"), (VAL_DIR, "Val")]:
        total = 0
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.png")))
                total += count
        print(f"{split_name}: {total} images")


if __name__ == "__main__":
    main()
