# Test Suite Specification: Virtual Edge-Node v5.2

## 0. Test Philosophy & Coverage Goals

The goal of this test suite is to verify that the implementation:

1. **Conforms exactly to the v5.2 spec behaviorally** (state machine, timing, buffer logic, power model, DSP pipeline).
2. **Matches all shape and range contracts** (audio chunks, spectrograms, ONNX inputs/outputs, telemetry, summaries).
3. **Handles all defined edge cases and error conditions** without divergence from the spec.

If all tests described here pass, you should be confident that:

* The state machine and tick loop behave as specified under all normal and edge conditions.
* The physics model (battery, voltage, blindness) is implemented correctly.
* The DSP pipeline strictly produces the expected shapes and normalizations.
* Telemetry and mission summary outputs are correct and internally consistent.
* Error handling matches the specified behavior.

---

## 1. Configuration & Validation Tests (`config.py`, `validate_runtime_config`)

### 1.1 Config field sanity

**Test C1: Config constants match spec**

* **What:** Assert that all config constants are set to the spec values:

  * `SAMPLE_RATE == 16000`
  * `CHUNK_DURATION == 1.0`
  * `CONTEXT_WINDOW == 3.0`
  * `N_FFT == 1024`
  * `HOP_LENGTH == 214`
  * `N_MELS == TIME_STEPS == 224`
  * `BATTERY_CAPACITY_MAH == 10000`
  * `POWER_CONSUMPTION["LISTENING"] == 120`, `"TRANSMIT" == 600`, `"SLEEP" == 80`
  * `TX_DURATION_TICKS == 10`
  * `WAKE_THRESHOLD == 0.05`
  * `CONFIDENCE_THRESHOLD == 0.85`
  * `TX_COOLDOWN_SEC == 300`
  * `TX_COOLDOWN_TICKS == 300`
* **Why:** Ensures implementation hasn’t drifted on core parameters that everything else depends on (Sections 3 & 5).

### 1.2 Runtime validation behavior

**Test C2: `validate_runtime_config` passes for valid setup**

* Create:

  * a dummy ONNX file at `MODEL_PATH`,
  * an existing directory at `SIM_INPUT_DIR`,
  * a writable directory for `LOG_FILE`.
* Call `validate_runtime_config()` and assert **no exception**.
* **Why:** Confirms the validation function is wired to the correct paths and does not falsely fail (Section 7.3).

**Test C3: `validate_runtime_config` fails on missing model**

* Remove / rename the file at `MODEL_PATH`.
* Expect `AssertionError` with message containing “Model not found”.
* **Why:** Matches fatal error spec for model missing (7.2).

**Test C4: `validate_runtime_config` fails on missing input dir**

* Make `SIM_INPUT_DIR` non-existent.
* Expect `AssertionError` with message containing “Input directory not found”.
* **Why:** Matches spec for fatal input path misconfig.

---

## 2. Environment / Audio Iterator Tests (`hardware_sim.py` / environment)

These tests assume you have a class like `AudioEnvironment` that backs `env.read_chunk()`.

### 2.1 Sample rate & mono conversion

**Test E1: Resampling to 16 kHz**

* Create a short test file: e.g. 0.1 s tone at **8 kHz**.
* Place it in `./data/sim_input`.
* Call `read_chunk()` repeatedly until EOF.
* Assert:

  * returned chunk(s) are `float32` of length exactly 16000 per tick,
  * the effective frequency content is what you expect after resampling (rough sanity check).
* **Why:** Ensures “Sample Rate: must be 16 kHz, resample otherwise” is actually honored (Module A).

**Test E2: Stereo to mono averaging**

* Create a stereo WAV: left = sine(440 Hz), right = sine(880 Hz).
* Ensure sample rate is 16 kHz.
* After `read_chunk()`, verify:

  * shape is `(16000,)` (mono),
  * samples ≈ `(left + right) / 2` within a small tolerance.
* **Why:** Tests the `(left + right) / 2` behavior (Module A).

### 2.2 Partial chunk zero-padding

**Test E3: Final partial chunk is zero-padded**

* Create a mono 16 kHz file with length 1.5 s (24,000 samples).
* First `read_chunk()` → 16,000 samples (original).
* Second `read_chunk()`:

  * returned length = 16,000 samples,
  * first 8,000 samples equal to the remaining audio,
  * last 8,000 samples are exactly zeros.
* Third `read_chunk()` → returns `None`.
* **Why:** Directly tests the “Partial Chunks” rule (Module A).

### 2.3 Cross-file continuity

**Test E4: Multi-file continuity and EOF**

* Create two files in `sim_input`:

  * `a.wav`: 1.0 s (16,000 samples) of constant value `+0.5`.
  * `b.wav`: 2.0 s (32,000 samples) of constant value `-0.5`.
* Sorted order ensures `a.wav` then `b.wav`.
* Call `read_chunk()` four times:

  * 1st: 16k samples of +0.5
  * 2nd: 16k samples of -0.5 (first half of `b.wav`)
  * 3rd: 16k samples of -0.5 (second half of `b.wav`)
  * 4th: `None` (EOF).
* **Why:** Verifies “Multi-file Mode” and that files are treated as one continuous stream (Module A).

---

## 3. DSP Pipeline & RMS Gating (`dsp_pipeline.py`)

### 3.1 RMS on raw, normalization logic

**Test D1: RMS uses raw audio, not normalized**

* Construct `raw_chunk` with max amplitude 0.1 and RMS known analytically (e.g., constant 0.1).
* Call your pipeline’s RMS function and normalization function separately:

  * `rms_raw == 0.1` within tolerance.
  * `normalized_chunk` has max ≈ 1.0.
* Confirm that `rms_raw` does *not* change after normalization.
* **Why:** Matches Section C.1: RMS on raw, normalization separate.

**Test D2: Silence protection threshold**

* Case A: `raw_chunk` all zeros.

  * `max_val = 0`, `normalized_chunk` all zeros.
* Case B: `raw_chunk` with very small values (e.g., `1e-7`).

  * `max_val < 1e-6` → treat as silence → `normalized_chunk` all zeros.
* Case C: `raw_chunk` with `max_val = 1e-5`.

  * `normalized_chunk` max must be 1.0.
* **Why:** Validates the `1e-6` guard and ensures no NaNs or weird values (4.1).

### 3.2 WAKE_THRESHOLD behavior

**Test D3: RMS gating threshold behavior**

* Build three chunks (all 1 s, 16k samples):

  * `chunk_low`: constant amplitude 0.04.
  * `chunk_edge`: constant 0.05.
  * `chunk_high`: constant 0.06.
* Compute `rms_raw` for each:

  * `chunk_low`: `rms_raw < WAKE_THRESHOLD`.
  * `chunk_edge`: `rms_raw == WAKE_THRESHOLD` (within tolerance).
  * `chunk_high`: `rms_raw > WAKE_THRESHOLD`.
* Use them in the state machine tests (Section 4) to verify:

  * `chunk_low` & `chunk_edge` do **not** trigger inference,
  * `chunk_high` does trigger inference when buffer has 3 chunks.
* **Why:** Ensures strict `>` semantics as per state table.

### 3.3 Spectrogram shape and parameters

**Test D4: Spectrogram shape and frame count**

* Construct a 3 s buffer (48,000 samples) of any non-trivial normalized signal.
* Call `generate_spectrogram(buffer)` (or equivalent).
* Before padding:

  * If you expose the raw mel spectrogram, assert `S.shape == (224, 220)` for `center=False`.
* After padding:

  * Final normalized tensor shape `[3, 224, 224]` (channels, height, width).
  * Values ∈ `[0.0, 1.0]`.
* **Why:** Directly enforces Section 4.2.2 & 4.2.4.

**Test D5: Log-mel normalization monotonicity**

* Ensure that:

  * `S_db.max()` maps to a normalized value ≈ 1.0.
  * `S_db.min()` maps to ≈ 0.0.
  * There are no NaNs or infs in `S_db` or `S_norm`.
* **Why:** Validates `power_to_db` + min/max scaling behavior (4.2.3).

### 3.4 ONNX input contract

**Test D6: ONNX input batch tensor**

* Feed a known buffer through the full DSP pipeline and preparation for ONNX.
* Confirm:

  * final input tensor has shape `(1, 3, 224, 224)`,
  * dtype is `float32`,
  * value range `[0.0, 1.0]`.
* **Why:** Matches Module C and 4.2.5.

---

## 4. State Machine & Tick Loop Tests (`main.py` / state logic)

Here we want to exercise the **complete** tick loop with controlled audio and a stub model to control outputs.

### 4.1 Basic buffer & inference behavior

**Test S1: No inference before buffer has 3 chunks**

* Use a stub environment that returns a high-RMS chunk each tick (> threshold).
* Use a stub model that would always yield `conf = 0.99`, `class_id = 1` if called.
* Run 2 ticks:

  * Assert state stays `LISTENING`.
  * Assert `run_model()` has not been called.
* On tick 3:

  * Assert `run_model()` is called exactly once.
  * Assert state transitions to `TRANSMIT`.
* **Why:** Enforces the “3-tick warmup” and state table row for `LISTENING` → `INFERENCE`.

**Test S2: Sliding window when RMS below threshold**

* Use an environment that returns 4 consecutive low-RMS chunks (< WAKE_THRESHOLD).
* Ensure buffer logic behaves as:

  * After 3 ticks, buffer length == 3.
  * On each subsequent tick, buffer length remains 3, the oldest chunk is dropped, the newest added.
  * No inference is triggered.
* **Why:** Tests “Buffer == 3 AND RMS ≤ 0.05 → LISTENING, pop oldest, push new chunk.”

### 4.2 Detection path and blind gap

**Test S3: Full detection cycle timing**

* Setup:

  * Stub env: 3 high-RMS chunks, then continue high-RMS forever.
  * Stub model: always returns `(class_id=1 [VESSEL], conf=0.99)` when called.
* Run ticks and track state per tick:

  * Tick 0–1: `LISTENING`, buffer < 3.
  * Tick 2: `LISTENING` → `INFERENCE` → `TRANSMIT` within same tick.

    * After tick 2:

      * state == `TRANSMIT`
      * `tx_timer == TX_DURATION_TICKS` (10).
  * Ticks 3–12: state remains `TRANSMIT`, tx_timer decrements, no model calls, buffer stays empty.
  * At end of tick 12: tx_timer reaches 0, state becomes `SLEEP`, `sleep_timer == 300`.
  * Ticks 13–312: state `SLEEP`, sleep_timer decrements; no buffer modifications (until exit), no model calls.
  * At end of tick 312: state transitions to `LISTENING`, buffer is cleared.
  * Ticks 313–315: buffer refills 3 chunks, next detection triggers earliest at tick 315.
* **Why:** Validates:

  * exact TX duration (10 ticks),
  * exact SLEEP duration (300 ticks),
  * buffer clearing semantics,
  * earliest possible next inference timing (3+310=313 ticks after detection tick, with inference on 3rd new chunk).

### 4.3 Non-detection inference

**Test S4: Background or low-confidence path**

* Stub model returns `class_id=0` or `conf=0.5` when called.
* Use high-RMS chunks (so inference is triggered once buffer full).
* On first inference:

  * Assert `state` returns to `LISTENING`,
  * `buffer` is unchanged (length still 3),
  * `TRANSMIT` is never entered, no tx_timer set.
* **Why:** Tests the negative inference branch in the state table.

### 4.4 Blindness & buffer invariants

**Test S5: No inference during TRANSMIT or SLEEP, env pointer advances**

* Instrument:

  * Count of `env.read_chunk()` calls.
  * Count of `run_model()` calls.
* Scenario: same as S3.
* Verify:

  * During `TRANSMIT` & `SLEEP`, `env.read_chunk()` is still called every tick (pointer advances).
  * `run_model()` is never called when state is `TRANSMIT` or `SLEEP`.
  * `buffer` length remains 0 during entire blind period.
* **Why:** Confirms “Blindness Logic” and buffer flush rules.

### 4.5 EOF behavior

**Test S6: EOF in LISTENING**

* Environment with finite duration such that EOF occurs while in LISTENING (buffer <3 or =3).
* On the tick where `env.read_chunk()` returns `None`:

  * `enter_shutdown("AUDIO_EXHAUSTED")` is called.
  * Telemetry line `MISSION_END` is logged with `ts == tick_count`, `batt_v == current voltage`.
  * Mission summary is printed.
* **Why:** Enforces the shutdown path for AUDIO_EXHAUSTED (Module A, Shutdown Handling).

**Test S7: EOF after blind period (if you choose that semantics)**

Depending on how you decide to handle EOF in TRANSMIT/SLEEP, either:

* **A:** You implement immediate shutdown on EOF even in blind states → Test that `env.read_chunk()` returning `None` while in TX/SLEEP also calls `enter_shutdown("AUDIO_EXHAUSTED")`.

**OR**

* **B (current tick contract):** You only shut down when you next re-enter LISTENING and try to read. In that case, test:

  * Environment returns `None` starting at some tick during `SLEEP`.
  * After SLEEP completes and state returns to LISTENING, first call to `env.read_chunk()` returns `None` and triggers shutdown.
  * No further ticks run after shutdown.

* **Why:** Ensures your chosen EOF semantics are actually implemented; prevents subtle infinite loops at EOF.

---

## 5. Battery & Power Model Tests (`hardware_sim.py` / power logic)

### 5.1 Per-tick consumption & clamping

**Test B1: Single-tick consumption per state**

* For each state (`LISTENING`, `TRANSMIT`, `SLEEP`):

  * Set `current_capacity_mah = BATTERY_CAPACITY_MAH`.
  * Call `consume_power(state)` once.
  * Assert:

    * `current_capacity_mah == BATTERY_CAPACITY_MAH - POWER_CONSUMPTION[state] * (CHUNK_DURATION / 3600.0)`.
* **Why:** Checks the power formula (Section 3).

**Test B2: Capacity clamping at 0**

* Set `current_capacity_mah` to a small value (e.g. 1 mAh).
* Call `consume_power("TRANSMIT")`.
* Assert `current_capacity_mah >= 0` and not negative.
* After calling `update_voltage()`, assert `voltage == 3.0` (or very close).
* **Why:** Validates clamping and voltage mapping at empty battery.

### 5.2 Voltage mapping and shutdown threshold

**Test B3: Voltage ranges with known capacities**

* Set:

  * full: `current_capacity_mah = BATTERY_CAPACITY_MAH` → `voltage ≈ 4.2`.
  * mid: `current_capacity_mah = BATTERY_CAPACITY_MAH / 2` → `voltage ≈ 3.6`.
  * empty: `current_capacity_mah = 0` → `voltage ≈ 3.0`.
* Asserts within small tolerances.
* **Why:** Confirms linear mapping 3.0–4.2 V.

**Test B4: Low-battery shutdown trigger**

* Choose initial capacity so that after one `consume_power(state)` + `update_voltage()`, `voltage` crosses from >3.2 V to ≤3.2 V.
* Run one tick and assert:

  * `enter_shutdown("LOW_BATTERY")` is called.
  * Telemetry includes `LOW_BATTERY_SHUTDOWN`.
* **Why:** Tests low-voltage cutoff behavior.

---

## 6. Telemetry & Logging Tests (`comms.py` + integration points)

### 6.1 Detection events

**Test T1: Detection event contents**

* Use the detection scenario from S3 with known detection tick (e.g., tick 2).
* Ensure `log_detection` writes a telemetry line with:

  * `ts == tick_count` at moment of logging,
  * `batt_v` equal to current voltage,
  * `event` == `"VESSEL"` or `"CETACEAN"` matching class_id,
  * `conf` and `rms` values equal to the stubbed model output and computed RMS.
* Ensure the log is valid NDJSON (one JSON per line, parseable).
* **Why:** Verifies Section 6 event schema for normal detections.

### 6.2 Shutdown events

**Test T2: Mission end logging**

* Trigger `enter_shutdown("AUDIO_EXHAUSTED")` using Test S6 setup.
* After shutdown, read the `mission_log.jsonl`:

  * Find a log line with `event == "MISSION_END"`.
  * `ts` matches final `tick_count`.
  * `batt_v` matches final voltage.
* **Why:** Confirms `MISSION_END` is logged exactly once at termination.

**Test T3: Low battery shutdown logging**

* Trigger `enter_shutdown("LOW_BATTERY")` via B4.
* Confirm log contains `event == "LOW_BATTERY_SHUTDOWN"` with correct `ts` and `batt_v`.
* **Why:** Matches the spec’s shutdown logging behavior.

---

## 7. Mission Summary Tests (`print_mission_summary`)

These can be integration tests that capture stdout or a function that returns the formatted string.

### 7.1 Summary formatting and fields

**Test M1: Summary fields present**

* After running a short controlled simulation (e.g., S3 scenario, limited ticks), capture the mission summary.
* Assert it contains lines with:

  * `Termination Reason : ...`
  * `Total Runtime      : ...`
  * `Final Battery      : ...`
  * `DETECTIONS` section with Vessel/Cetacean counts.
  * `COVERAGE` section with:

    * `Active Time`
    * `Blind Time`
    * `Inference Runs`
  * `POWER CONSUMPTION` with `Total mAh Used`, `Avg Current`.
* **Why:** Validates that the printed summary adheres to the specified template (Section 8).

### 7.2 Internal consistency of summary metrics

**Test M2: Coverage + power summary consistent with logs**

* For a small deterministic scenario (e.g., 1 detection, then immediate EOF):

  * Keep:

    * `tick_count` from main loop.
    * Counters: `listening_ticks`, `blind_ticks`, `inference_runs`.
    * `current_capacity_mah` changes over time.
  * After the run:

    * Assert: `Active Time == tick_count - Blind Time`.
    * Assert: `Blind Time == ticks in TRANSMIT + ticks in SLEEP`.
    * Compute `Total mAh Used` from initial minus final capacity; ensure it matches the number printed in summary within rounding tolerance.
    * Compute `Avg Current` = `Total mAh Used * 3600 / tick_count` and compare to printed value.
* **Why:** Ensures the mission summary is not just formatted correctly, but numerically correct.

---

## 8. Error Handling Tests (Recoverable & Fatal)

### 8.1 Recoverable errors

**Test E5: Corrupted audio chunk**

* Mock `env.read_chunk()` to sometimes raise an exception or return an invalid buffer.
* Verify:

  * Implementation logs a warning,
  * Skips that chunk and continues with next,
  * Does not crash the simulation.
* **Why:** Tests the “Corrupted audio chunk” path (Section 7.1).

**Test E6: Inference failure**

* Mock `run_model()` to throw an exception.
* Verify:

  * Error is logged,
  * State returns to `LISTENING`,
  * No `TRANSMIT` is triggered,
  * Main loop continues.
* **Why:** Matches “ONNX inference failure → log error, return to LISTENING” (7.1).

### 8.2 Fatal errors

**Test E7: No audio files fatal**

* Make sure `./data/sim_input` is empty.
* Start simulation:

  * Expect immediate fatal error with exit code 1 or equivalent.
  * Error log contains message about “No audio files in ./data/sim_input”.
* **Why:** Tests that spec-specified fatal misconfig is enforced (7.2).

**Test E8: Log directory not writable**

* Set log directory to a non-writable path.
* Start simulation.
* Expect fatal log/warning printed to stderr and process exits with code 1.
* **Why:** Enforces “Log directory not writable → fatal” behavior.

---

## 9. End-to-End Scenario Tests

Finally, put it all together with realistic-ish signals.

### 9.1 Single detection & full cooldown

**Test EE1: Realistic sample causing a single detection**

* Use:

  * A low-noise background WAV that generates low RMS (below threshold).
  * Embedded segment (3–10 s) of louder synthetic “vessel” noise that triggers detection once.
* Simulation parameters:

  * Run until EOF.
* Validate:

  * Exactly 1 detection event in logs with `event == "VESSEL"`.
  * TX and SLEEP durations match spec (10 + 300 ticks).
  * Coverage metrics in summary reflect one blind gap (~310 ticks worth).
  * Battery usage and voltage remain reasonable.
* **Why:** Confirms the whole stack, from audio → DSP → model → state machine → telemetry → summary, on a “happy path.”

### 9.2 No detection mission

**Test EE2: All-below-threshold mission**

* Use a WAV of very low amplitude noise so `rms_raw < WAKE_THRESHOLD` across the entire mission.
* Run until EOF.
* Validate:

  * No calls to `run_model()` (or zero inference_runs).
  * No `VESSEL`/`CETACEAN` events in log.
  * No time spent in TRANSMIT or SLEEP.
  * All ticks counted as Active Time in summary.
* **Why:** Validates that RMS gating alone can suppress inferences and that the system never enters blind states unnecessarily.

### 9.3 Multiple detections & battery drain

**Test EE3: Multi-detection scenario**

* Construct audio with multiple high-RMS segments separated by at least 3 + 310 seconds so each can trigger a separate detection.
* Run until low battery or EOF.
* Validate:

  * Number of detections == number of high-RMS segments (within expected constraints).
  * For each detection:

    * A corresponding TX+SLEEP segment appears in state history.
    * Telemetry includes a detection event at the correct tick.
  * Battery capacity and voltage monotonically decrease; final battery makes sense given the number of detections and blind gaps.
* **Why:** Stress-tests repeated detection and ensures the system doesn’t get stuck in any state.

---

## 10. Invariants & Property-Based Checks (Optional but Strong)

These can be done as lightweight property tests with random or semi-random sequences:

* **Invariant P1:** `len(buffer) <= 3` always.
* **Invariant P2:** `run_model()` is called only when `len(buffer) == 3` and state is transitioning from LISTENING to INFERENCE.
* **Invariant P3:** `tx_timer` is in `[0, TX_DURATION_TICKS]`; `sleep_timer` in `[0, TX_COOLDOWN_TICKS]`.
* **Invariant P4:** At any time, `voltage ∈ [3.0, 4.2]`.
* **Invariant P5:** `ActiveTime + BlindTime == totalTicks` in summary.

Run a randomized simulation with stubbed environment/model for several thousand ticks and assert these invariants on every tick.

---

If you build a test suite that covers all of the above, you’re not just checking “a few functions”—you’re:

* Verifying **every edge case** explicitly called out in v5.2.
* Locking in **all shape/range contracts** for audio, spectrogram, and model IO.
* Ensuring that the **state machine, battery model, telemetry, and summaries** are consistent and correct.

Passing this suite doesn’t *mathematically* prove the implementation is bug-free, but for a system of this size and complexity, it’s about as close to “confidence that it’s working as intended” as you’re likely to get.
