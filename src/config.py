# config.py
"""
Configuration constants for the marine sensor node simulation.
All values are defined per the Master Engineering Specification v5.2.
"""

# --- AUDIO SETTINGS ---
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0       # Seconds per tick
CONTEXT_WINDOW = 3.0       # Seconds sent to ML
N_FFT = 1024
HOP_LENGTH = 214           # Tuned for ~220 frames (padded to 224)
N_MELS = 224               # Image Height
TIME_STEPS = 224           # Target Image Width

# --- PHYSICS MODEL ---
BATTERY_CAPACITY_MAH = 10000
POWER_CONSUMPTION = {
    "LISTENING": 120,      # Includes inference (amortized)
    "TRANSMIT": 600,
    "SLEEP": 80
}
TX_DURATION_TICKS = 10     # Transmission blocks sensor for 10s

# --- LOGIC THRESHOLDS ---
WAKE_THRESHOLD = 0.05      # RMS on raw (pre-normalized) audio
CONFIDENCE_THRESHOLD = 0.85
TX_COOLDOWN_SEC = 300      # 5 Minute debounce
TX_COOLDOWN_TICKS = int(TX_COOLDOWN_SEC / CHUNK_DURATION)  # 300 ticks

# --- PATHS ---
DATA_DIR = "./data"
SIM_INPUT_DIR = "./data/sim_input"
LOG_FILE = "./logs/mission_log.jsonl"
MODEL_PATH = "./models/classifier.onnx"

# --- CLASS DEFINITIONS ---
CLASS_NAMES = {
    0: "Background",
    1: "Vessel",
    2: "Cetacean"
}

# --- DERIVED CONSTANTS ---
# Used by both simulation and dashboard to ensure consistent audio sizing
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)  # 16000 samples at default config

# --- DASHBOARD INSTRUMENTATION ---
TICK_STATE_FILE = "./logs/tick_state.jsonl"
LOG_ARCHIVE_DIR = "./logs/archive"
DASHBOARD_AUDIO_ENABLED = True  # Set False to disable audio serialization
