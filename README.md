# Virtual Edge-Node for Acoustic Anomaly Detection
**Target Platform:** Raspberry Pi Zero 2 W + RockBLOCK 9603
**Simulation Mode:** Discrete Time-Step (1 Tick = 1.0 Second)

---

## 1. Executive Summary
This project simulates an autonomous marine sensor node. The system processes continuous hydrophone data to detect vessel traffic and marine mammals. It operates under a "Power-First" architecture, where reporting data induces blindness to new data (the Coverage Gap).

**Key Constraints:**
*   **Voltage:** 1S LiPo (3.0V - 4.2V).
*   **Blind Spots:** Transmission blocks sensor input; Cool-down periods enforce silence.
*   **Telemetry:** Bandwidth limited to <340 Bytes per event.

---

## 2. System Architecture

### Module A: The Environment (Physics Engine)
*   **Input Handling:** Scans `./data/sim_input` for audio files (`.wav` or `.flac`).
    *   **File Selection:** Files are sorted alphabetically; the **first file** is processed.
    *   **Multi-file Mode:** After the first file is exhausted, continue to the next file in sorted order until all files are processed.
    *   **Sample Rate:** Input files **must be 16kHz**. If a file has a different sample rate, resample to 16kHz using `librosa.resample()` before processing.
    *   **Channels:** If stereo, convert to mono by averaging channels: `mono = (left + right) / 2`.
    *   **Partial Chunks:** If fewer than 16,000 samples remain at end of file:
        *   **Zero-pad** the final chunk to 16,000 samples.
        *   This partial chunk is treated as a valid tick and processed normally.
    *   **Cross-file Continuity:** The 3-second context window **may span file boundaries**. Files are treated as one continuous audio stream (e.g., last 2s of file A + first 1s of file B form a valid buffer).
*   **End-of-Stream:** When all audio files are exhausted, `env.read_chunk()` returns `None`. The main loop detects this and calls `enter_shutdown("AUDIO_EXHAUSTED")`.
    *   **Note:** EOF is only checked when actively reading audio (LISTENING state). If EOF occurs during TRANSMIT/SLEEP, the system completes those states first, then shuts down on the next attempted read after returning to LISTENING.
*   **Blindness Logic:** During `TRANSMIT` or `SLEEP` states, the simulation clock ticks and the audio file pointer advances, but **data is discarded** (not passed to the firmware).

### Module B: The Firmware (State Machine)
*   **Initial State:** `LISTENING` with an empty buffer and full battery (100% capacity).
*   **Buffer Implementation:** `collections.deque(maxlen=3)` storing **1-second Numpy arrays**.
*   **Buffer Flushing:**
    *   On entering `TRANSMIT` (after a detection), the buffer is **cleared**.
    *   On returning from `SLEEP` to `LISTENING`, the buffer is **cleared** (defensive, ensures fresh 3-tick warmup).
    *   After any buffer clear, the system must wait 3 ticks to refill before inference can trigger.

#### State Machine Diagram
```
                         ┌─────────────────────────────────────────────────────────┐
                         │                      [ANY STATE]                        │
                         │                           │                             │
                         │              V ≤ 3.2V (Low Battery)                     │
                         │                           ▼                             │
                         │                     ┌──────────┐                        │
                         │                     │ SHUTDOWN │ ──► sys.exit(0)        │
                         │                     └──────────┘                        │
                         └─────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │  NORMAL OPERATION                                                            │
    │                                                                              │
    │   ┌───────────┐                                                              │
    │   │   START   │                                                              │
    │   └─────┬─────┘                                                              │
    │         │                                                                    │
    │         ▼                                                                    │
    │   ┌───────────┐  Buffer < 3 chunks    ┌───────────┐                          │
    │   │ LISTENING │◄──────────────────────│ LISTENING │◄─────────────────┐       │
    │   └─────┬─────┘  (stay, accumulate)   └───────────┘                  │       │
    │         │                                                            │       │
    │         │ Buffer == 3 chunks AND RMS > 0.05                          │       │
    │         ▼                                                            │       │
    │   ┌───────────┐                                                      │       │
    │   │ INFERENCE │                                                      │       │
    │   └─────┬─────┘                                                      │       │
    │         │                                                            │       │
    │         ├─── Conf < 0.85 OR Class == Background ─────────────────────┘       │
    │         │                                                                    │
    │         │ Conf ≥ 0.85 AND Class ∈ {Vessel, Cetacean}                         │
    │         ▼                                                                    │
    │   ┌───────────┐                                                              │
    │   │ TRANSMIT  │  (10 ticks, buffer cleared, audio discarded)                 │
    │   └─────┬─────┘                                                              │
    │         │                                                                    │
    │         │ TX Complete                                                        │
    │         ▼                                                                    │
    │   ┌───────────┐                                                              │
    │   │   SLEEP   │  (300 ticks, audio discarded)                                │
    │   └─────┬─────┘                                                              │
    │         │                                                                    │
    │         │ Cooldown Complete                                                  │
    │         ▼                                                                    │
    │   ┌───────────┐                                                              │
    │   │ LISTENING │  (buffer empty, must refill 3 chunks before next inference)  │
    │   └───────────┘                                                              │
    │                                                                              │
    └──────────────────────────────────────────────────────────────────────────────┘
```

#### State Transition Table
| Current State | Condition | Next State | Action |
| :--- | :--- | :--- | :--- |
| LISTENING | Buffer < 3 chunks | LISTENING | Accumulate audio chunk |
| LISTENING | Buffer == 3 AND RMS ≤ 0.05 | LISTENING | Pop oldest, push new chunk |
| LISTENING | Buffer == 3 AND RMS > 0.05 | INFERENCE | Run ML model |
| INFERENCE | Conf < 0.85 OR Class == 0 | LISTENING | Continue monitoring |
| INFERENCE | Conf ≥ 0.85 AND Class ∈ {1,2} | TRANSMIT | Clear buffer, start TX |
| TRANSMIT | TX timer < 10 | TRANSMIT | Decrement timer, discard audio |
| TRANSMIT | TX timer == 0 | SLEEP | Log event, start cooldown |
| SLEEP | Cooldown timer < 300 | SLEEP | Decrement timer, discard audio |
| SLEEP | Cooldown timer == 0 | LISTENING | Resume with empty buffer |
| ANY | V ≤ 3.2V | SHUTDOWN | Flush logs, terminate |
| LISTENING | Audio stream exhausted | SHUTDOWN | Flush logs, terminate |

#### Tick Loop Contract (Exact Ordering)

Each simulation tick executes the following steps **in order**:

```python
def tick():
    global state, tick_count, buffer, tx_timer, sleep_timer

    # 1. CONSUME POWER for current state (before any state change)
    consume_power(state)

    # 2. UPDATE VOLTAGE after consumption
    update_voltage()

    # 3. CHECK LOW-BATTERY (triggers SHUTDOWN if V <= 3.2)
    if voltage <= 3.2:
        enter_shutdown("LOW_BATTERY")
        return

    # 4. READ AUDIO from environment (or discard if blind)
    if state in (State.TRANSMIT, State.SLEEP):
        _ = env.read_chunk()  # Advance pointer, discard data
        raw_chunk = None
    else:
        raw_chunk = env.read_chunk()  # Returns None if EOF
        if raw_chunk is None:
            enter_shutdown("AUDIO_EXHAUSTED")
            return

    # 5. STATE-SPECIFIC LOGIC
    if state == State.LISTENING:
        rms_raw = compute_rms(raw_chunk)
        normalized = normalize(raw_chunk)
        buffer.append(normalized)

        if len(buffer) == 3 and rms_raw > WAKE_THRESHOLD:
            state = State.INFERENCE
            # Fall through to INFERENCE logic THIS tick

    if state == State.INFERENCE:
        spectrogram = generate_spectrogram(buffer)
        class_id, confidence = run_model(spectrogram)

        if confidence >= CONFIDENCE_THRESHOLD and class_id in (1, 2):
            log_detection(class_id, confidence, rms_raw)
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
            buffer.clear()  # Ensure buffer is empty on re-entry
            state = State.LISTENING

    # 6. INCREMENT TICK COUNTER
    tick_count += 1
```

**Key Ordering Notes:**
*   Power is consumed at the **start** of each tick based on the state at tick entry.
*   Low-battery check happens **after** power consumption (catches the tick that depletes battery).
*   LISTENING → INFERENCE transition and inference execution happen within the **same tick**.
*   Timer decrements happen **during** the tick; transition occurs when timer reaches 0.
*   `buffer.clear()` is called on entry to TRANSMIT (after detection) AND on exit from SLEEP (defensive).

#### Shutdown Handling

```python
def enter_shutdown(reason):
    """Handle both LOW_BATTERY and AUDIO_EXHAUSTED termination."""
    event_type = "LOW_BATTERY_SHUTDOWN" if reason == "LOW_BATTERY" else "MISSION_END"
    log_event({"ts": tick_count, "batt_v": voltage, "event": event_type})
    flush_logs()
    print_mission_summary(reason)
    sys.exit(0)
```

All shutdown paths (low battery, audio exhausted) are handled identically by the main loop—**not** by the environment module.

### Module C: The Intelligence (Inference Engine)
*   **Model:** MobileNetV2 (Quantized ONNX).
*   **Input:** 224x224 Mel-Spectrogram (Stacked 3-channel).
*   **Output:** Softmax probability vector `[p_background, p_vessel, p_cetacean]`.
*   **Confidence:** `max(output)` — the highest probability value.
*   **Predicted Class:** `argmax(output)` — index of highest probability.
*   **Classes:** `0: Background`, `1: Vessel`, `2: Cetacean`.

### Module C.1: RMS Calculation (Power Gating)
*   **Scope:** RMS is calculated on the **most recent 1-second chunk only** (not the full 3-second buffer).
*   **CRITICAL:** RMS must be computed on the **raw (pre-normalized) audio** to reflect true acoustic amplitude.
*   **Formula:** `rms_raw = np.sqrt(np.mean(raw_chunk ** 2))` where `raw_chunk` is Float32 audio before normalization.
*   **Amplitude Convention:** All audio is converted to float32 in the range `[-1.0, 1.0]`, where ±1.0 corresponds to full-scale amplitude (e.g., from `soundfile.read()` or `librosa.load()`). `WAKE_THRESHOLD = 0.05` means ~5% of full-scale RMS.
*   **Timing:** Calculated every tick in LISTENING state, **before** normalization is applied.
*   **Trigger Logic:** Inference runs when `len(buffer) == 3 AND rms_raw > WAKE_THRESHOLD`.
*   **Rationale:** Per-chunk normalization rescales all non-silent audio to similar amplitude ranges, which would cause RMS to always exceed threshold. Using raw RMS preserves true amplitude gating for power savings.

### Module D: Telemetry (Comms Interface)
*   **Format:** NDJSON (Newline Delimited JSON).
*   **File:** `logs/mission_log.jsonl` (Append mode).

---

## 3. Hardware Definitions (The Physics Model)

### Power Consumption & Timing Table

| State | Description | Current | Voltage | Duration (Ticks) | Audio Input? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LISTENING** | Buffer fill + RMS Check + Inference | **120 mA** | 3.7V | 1 | **Yes** |
| **TRANSMIT** | Modem Handshake + TX | **600 mA** | 3.7V | **10 (Blocking)** | **NO** |
| **SLEEP** | Post-TX Cool-down | **80 mA** | 3.7V | **300 (Timer)** | **NO** |

**Note on INFERENCE:** In this simulation, inference compute is treated as a **logical sub-step** within LISTENING ticks, not a separate power state. The 120 mA LISTENING current includes the amortized cost of occasional inference runs. This simplifies the power model while remaining realistic for the low-duty-cycle detection scenario.

*   **Total Blind Spot:** 310 seconds per detection event.

#### Battery Model Formulas
```python
# Per-tick capacity decrement (called at start of each tick):
def consume_power(state):
    global current_capacity_mah
    current_ma = POWER_CONSUMPTION[state]  # mA for this state
    delta_mah = current_ma * (CHUNK_DURATION / 3600.0)  # Convert to mAh
    current_capacity_mah = max(0, current_capacity_mah - delta_mah)

# Voltage calculation (called after capacity update):
def update_voltage():
    global voltage
    ratio = current_capacity_mah / BATTERY_CAPACITY_MAH  # 0.0 to 1.0
    voltage = 3.0 + 1.2 * ratio  # Range: 3.0V (empty) to 4.2V (full)
```

*   **Capacity Clamping:** Capacity is clamped to `[0, BATTERY_CAPACITY_MAH]`.
*   **Voltage Update Order:** Voltage is recalculated **after** consuming power for the current tick.
*   **Low-Voltage Cutoff:** When $V \leq 3.2V$ (approx. 17% capacity), the main loop calls `enter_shutdown("LOW_BATTERY")`. See Shutdown Handling in the Tick Loop Contract for details.

---

## 4. Signal Processing Pipeline

### 4.1 Ingest & Normalization
*   **Step 1:** Read 1.0 second of audio (16,000 samples).
*   **Step 2:** Convert to Float32 (`raw_chunk`).
*   **Step 3:** Compute RMS for power gating (on raw data):
    ```python
    rms_raw = np.sqrt(np.mean(raw_chunk ** 2))
    ```
*   **Step 4:** Normalize with silence protection (for model input):
    ```python
    max_val = np.max(np.abs(raw_chunk))
    if max_val > 1e-6:  # Avoid division by zero for silent audio
        normalized_chunk = raw_chunk / max_val
    else:
        normalized_chunk = np.zeros_like(raw_chunk)  # Treat as silence
    ```
*   **Step 5:** Push `normalized_chunk` to Deque (buffer stores normalized data for spectrogram).
*   **Step 6:** Use `rms_raw` for wake threshold comparison (see Module C.1).

### 4.2 Spectrogram Generation (Dynamic Padding)

*   **Input:** 3.0 Seconds (48,000 samples) from buffer, concatenated.

#### 4.2.1 STFT Parameters (Exact Specification)
```python
# Librosa call for reproducibility:
S = librosa.feature.melspectrogram(
    y=audio_3s,           # 48,000 samples, float32, normalized
    sr=16000,
    n_fft=1024,
    hop_length=214,
    n_mels=224,
    window='hann',        # Hann window function
    center=False,         # No padding; frame count matches formula exactly
    power=2.0,            # Power spectrogram (magnitude squared)
    fmin=20.0,            # Min frequency for mel filterbank (Hz)
    fmax=8000.0           # Max frequency for mel filterbank (Hz, Nyquist for 16kHz)
)
```

#### 4.2.2 Frame Count Math
*   Formula (with `center=False`): `frames = floor((n_samples - n_fft) / hop_length) + 1`
*   With `HOP_LENGTH = 214`: `floor((48000 - 1024) / 214) + 1 = 220` frames exactly.
*   Dynamic padding adds 4 frames to reach 224.

#### 4.2.3 Log-Mel Conversion & Normalization
```python
# Convert to log scale (dB)
S_db = librosa.power_to_db(S, ref=np.max)  # Range: [-80, 0] dB typical

# Normalize to [0, 1] for model input
S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
```

#### 4.2.4 Padding & Final Shape
1.  After mel-spectrogram: shape `[224, 220]` (mels × frames).
2.  Calculate `pad_width = 224 - 220 = 4`.
3.  Zero-pad on right (time axis): `np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant')`.
4.  Final shape: `[224, 224]`.
5.  Stack 3x for RGB channels: `[3, 224, 224]`.

#### 4.2.5 ONNX Model Input Convention
*   **Shape:** `[1, 3, 224, 224]` (batch, channels, height, width) — NCHW format.
*   **Dtype:** `float32`.
*   **Value Range:** `[0.0, 1.0]` (from normalization above).
*   **Channel Stacking:** All 3 channels contain identical spectrogram data (grayscale replicated to RGB).

---

## 5. Configuration (`config.py`)

```python
# config.py

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
```

---

## 6. Telemetry Data Structure
**Log Format:** NDJSON (One valid JSON object per line).

**Example Payload:**
```json
{"ts": 3605, "batt_v": 4.1, "event": "CETACEAN", "conf": 0.96, "rms": 0.12}
{"ts": 7210, "batt_v": 4.0, "event": "VESSEL", "conf": 0.89, "rms": 0.35}
```
*Note the voltage is now correctly modeled (3.0V - 4.2V range).*

**Telemetry Size Constraint:**
*   Maximum payload size: **340 bytes** per event (RockBLOCK SBD limit).
*   With this fixed schema, events are ~60 bytes—well under the limit.
*   The simulation does **not** implement runtime size enforcement or truncation logic.

**Event Types:**
| Event | Description |
| :--- | :--- |
| `VESSEL` | Vessel detected with confidence ≥ 0.85 |
| `CETACEAN` | Marine mammal detected with confidence ≥ 0.85 |
| `LOW_BATTERY_SHUTDOWN` | System shutdown due to low voltage |
| `MISSION_END` | Audio stream exhausted, normal termination |

**Timestamp Semantics:**
*   `ts`: Integer tick index, starting at `0` for the first tick.
*   Each tick represents exactly 1.0 second of simulation time.
*   Conversion: `wallclock_seconds = ts * CHUNK_DURATION`.

---

## 7. Error Handling

### 7.1 Recoverable Errors (Log and Continue)
| Error | Handling |
| :--- | :--- |
| Corrupted audio chunk | Log warning, skip chunk, continue with next |
| ONNX inference failure | Log error, return to LISTENING state |
| Single file read error | Log warning, skip to next file in queue |

### 7.2 Fatal Errors (Log and Terminate)
| Error | Handling |
| :--- | :--- |
| No audio files in `./data/sim_input` | Log error, `sys.exit(1)` |
| ONNX model file not found | Log error, `sys.exit(1)` |
| ONNX model load failure | Log error, `sys.exit(1)` |
| Log directory not writable | Print to stderr, `sys.exit(1)` |
| Invalid config values | Log error with specifics, `sys.exit(1)` |

### 7.3 Config Validation Rules

**Important:** Config validation should be called from `main.py` at runtime startup, NOT at module import time. This allows `train_model.py` to import config values without requiring the model file to exist yet.

```python
# In main.py, call this function before entering the main loop:
def validate_runtime_config():
    """Validate config values at simulation startup (not import time)."""
    import os
    from config import (SAMPLE_RATE, WAKE_THRESHOLD, CONFIDENCE_THRESHOLD,
                        TX_DURATION_TICKS, TX_COOLDOWN_SEC, BATTERY_CAPACITY_MAH,
                        MODEL_PATH, SIM_INPUT_DIR, LOG_FILE)

    assert SAMPLE_RATE > 0, "SAMPLE_RATE must be positive"
    assert 0.0 < WAKE_THRESHOLD < 1.0, "WAKE_THRESHOLD must be in (0, 1)"
    assert 0.0 < CONFIDENCE_THRESHOLD <= 1.0, "CONFIDENCE_THRESHOLD must be in (0, 1]"
    assert TX_DURATION_TICKS > 0, "TX_DURATION_TICKS must be positive"
    assert TX_COOLDOWN_SEC >= 0, "TX_COOLDOWN_SEC must be non-negative"
    assert BATTERY_CAPACITY_MAH > 0, "BATTERY_CAPACITY_MAH must be positive"
    assert os.path.isfile(MODEL_PATH), f"Model not found: {MODEL_PATH}"
    assert os.path.isdir(SIM_INPUT_DIR), f"Input directory not found: {SIM_INPUT_DIR}"
    assert os.access(os.path.dirname(LOG_FILE) or '.', os.W_OK), "Log directory not writable"
```

---

## 8. Mission Summary Format

When the simulation ends (audio exhausted or low battery), print the following to console:

```
================================================================================
                            MISSION SUMMARY
================================================================================
Termination Reason : {AUDIO_EXHAUSTED | LOW_BATTERY}
Total Runtime      : {ticks} ticks ({hours}h {minutes}m {seconds}s)
Final Battery      : {voltage:.2f}V ({percentage:.1f}%)

DETECTIONS:
  Vessels          : {count}
  Cetaceans        : {count}
  Total Events     : {count}

COVERAGE:
  Active Time      : {ticks} ticks ({percentage:.1f}%)  # Ticks where audio was processed
  Blind Time       : {ticks} ticks ({percentage:.1f}%)  # TRANSMIT + SLEEP
  Inference Runs   : {count}                            # Times model was invoked

POWER CONSUMPTION:
  Total mAh Used   : {mah:.1f} mAh
  Avg Current      : {avg_ma:.1f} mA
================================================================================
```

**Coverage Metric Definitions:**
*   **Active Time:** Count of ticks where audio was read and processed (not discarded). Since INFERENCE is a logical sub-step within LISTENING ticks, Active Time = total ticks − Blind Time.
*   **Blind Time:** Count of ticks spent in TRANSMIT or SLEEP states.
*   **Inference Runs:** Maintain a separate counter incremented each time `run_model()` is called.

---

## 10. Directory Structure

```text
/project_root
    /src
        config.py
        hardware_sim.py        # Battery logic + Audio File Iterator
        dsp_pipeline.py        # Buffer management + Spectrogram logic
        inference.py           # ONNX loading + Execution
        comms.py               # JSON formatting
        main.py                # While loop + State Machine
    /data
        /raw_synthetic         # Generated .wavs
        /processed_spectrograms
            /train
            /val
        /sim_input             # Place long NOAA .flac here
    /logs
        mission_log.jsonl      # Output telemetry
    /models
        classifier.onnx
    generate_dummy_data.py
    train_model.py
    requirements.txt
    README.md
```

---

## 11. To Run

```text
  1. uv venv && uv pip install -r requirements.txt -r requirements-training.txt -r requirements-test.txt -r requirements-dashboard.txt
  2. uv run python generate_dummy_data.py — creates training data
  3. uv run python train_model.py — trains and exports to models/classifier.onnx
  4. Place audio in data/sim_input/
  5. uv run python src/main.py — runs simulation
  6. uv run streamlit run dashboard/app.py 
```