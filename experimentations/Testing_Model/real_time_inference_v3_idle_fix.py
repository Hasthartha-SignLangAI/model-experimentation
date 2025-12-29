import time
import json
import numpy as np
import tensorflow as tf
import serial
from collections import deque, Counter

# =====================
# CONFIG (EDIT THESE)
# =====================
PORT = "/dev/tty.usbserial-0001"
BAUD = 115200

MODEL_DIR = "cnn_lstm_savedmodel"  # SavedModel folder
T = 512
F = 9

# --- Idle calibration & drift handling ---
CALIB_SECONDS = 3.0      # keep arm relaxed for first N seconds
K_START = 5.5            # raise if false triggers at idle (try 6..10)
K_END   = 3.0            # lower than K_START for hysteresis
COOLDOWN_FRAMES = 60     # ignore triggers after a gesture (~0.6s @100Hz)

# Optional: slowly adapt baseline during long idle (helps drift)
ADAPT_IDLE_BASELINE = True
IDLE_ADAPT_RATE = 0.01   # smaller = slower (0.005..0.02)
IDLE_ADAPT_MIN_FRAMES = 200  # wait ~2s idle before adapting

# --- Output behaviour ---
CONF_TH = 0.80              # minimum confidence to accept word
PRED_INTERVAL_FRAMES = 6    # run inference every N frames near end
END_QUIET_FRAMES = 35      # quiet frames to declare end (~0.2s)
MIN_GESTURE_FRAMES = 35
MAX_GESTURE_FRAMES = 300    # safety (~3s)

MIN_VOTE_FRAC = 0.60         # winner must have >=60% of votes
MIN_VOTES = 6                # need at least 6 predictions collected


# Energy smoothing
ENERGY_SMOOTH_W = 8

# =====================
# Load label map + scaler
# =====================
with open("label_map.json") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

with open("scaler_params.json") as f:
    sc = json.load(f)
mean = np.array(sc["mean"], dtype=np.float32)
scale = np.array(sc["scale"], dtype=np.float32)

# =====================
# Load SavedModel signature
# =====================
loaded = tf.saved_model.load(MODEL_DIR)
infer = loaded.signatures["serving_default"]

input_keys = list(infer.structured_input_signature[1].keys())
output_keys = list(infer.structured_outputs.keys())
if len(input_keys) != 1 or len(output_keys) != 1:
    print("⚠️ Signature keys not single. Inputs:", input_keys, "Outputs:", output_keys)
    raise SystemExit(1)

IN_KEY = input_keys[0]
OUT_KEY = output_keys[0]

print("Using SavedModel signature:")
print("  IN_KEY :", IN_KEY)
print("  OUT_KEY:", OUT_KEY)

# =====================
# Helpers
# =====================
def normalize(X: np.ndarray) -> np.ndarray:
    return (X - mean) / (scale + 1e-6)

def emg_energy(row9: np.ndarray) -> float:
    return float(np.sum(np.abs(row9[:3])))

def moving_avg_update(buf: deque, new_val: float) -> float:
    buf.append(new_val)
    return float(sum(buf) / len(buf))

def predict_from_buffer(buffer: deque) -> tuple[int, float]:
    X = np.array(buffer, dtype=np.float32)  # (T,9)
    X[:, :3] -= X[:, :3].mean(axis=0, keepdims=True)  # EMG DC removal
    X = normalize(X)
    X = X[np.newaxis, ...].astype(np.float32)          # (1,T,9)

    out = infer(**{IN_KEY: tf.constant(X)})
    probs = out[OUT_KEY].numpy()[0]                    # (10,)
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return pred, conf

# =====================
# Serial setup
# =====================
ser = serial.Serial(PORT, BAUD, timeout=1)
print("Listening on", PORT)

# ---- Idle calibration ----
print(f"Calibrating idle baseline for {CALIB_SECONDS}s... keep arm relaxed")
t0 = time.time()
calib_vals = []

while time.time() - t0 < CALIB_SECONDS:
    line = ser.readline().decode(errors="ignore").strip()
    if not line or "t_ms" in line:
        continue
    parts = line.split(",")
    if len(parts) != 10:
        continue
    try:
        vals = np.array(parts[1:], dtype=np.float32)
    except:
        continue
    calib_vals.append(emg_energy(vals))

calib_vals = np.array(calib_vals, dtype=np.float32)
idle_mean = float(np.mean(calib_vals))
idle_std  = float(np.std(calib_vals) + 1e-6)
print(f"Idle baseline: mean={idle_mean:.1f}, std={idle_std:.1f}")

# =====================
# State
# =====================
buffer = deque(maxlen=T)
energy_raw = deque(maxlen=ENERGY_SMOOTH_W)

in_gesture = False
quiet = 0
gesture_frames = 0

pred_counter = Counter()
conf_counter = Counter()
frames_since_last_pred = 0

cooldown = 0
idle_frames = 0  # for baseline adaptation

# =====================
# Main loop
# =====================
while True:
    line = ser.readline().decode(errors="ignore").strip()
    if not line or "t_ms" in line:
        continue

    parts = line.split(",")
    if len(parts) != 10:
        continue

    try:
        vals = np.array(parts[1:], dtype=np.float32)  # drop timestamp
    except:
        continue

    buffer.append(vals)
    if len(buffer) < T:
        continue

    # Smooth energy
    e = emg_energy(vals)
    e_s = moving_avg_update(energy_raw, e)

    # cooldown countdown
    if cooldown > 0:
        cooldown -= 1

    # thresholds from idle baseline (NOT ratio-based)
    start_th = idle_mean + K_START * idle_std
    end_th   = idle_mean + K_END   * idle_std

    # =====================
    # IDLE state
    # =====================
    if not in_gesture:
        # Optional baseline drift adaptation during long idle
        if ADAPT_IDLE_BASELINE and e_s < end_th:
            idle_frames += 1
            if idle_frames >= IDLE_ADAPT_MIN_FRAMES:
                # exponential moving average update of idle_mean
                idle_mean = (1 - IDLE_ADAPT_RATE) * idle_mean + IDLE_ADAPT_RATE * e_s
        else:
            idle_frames = 0

        # Start condition: must pass cooldown + exceed start threshold
        if cooldown == 0 and e_s > start_th:
            in_gesture = True
            quiet = 0
            gesture_frames = 0
            pred_counter.clear()
            conf_counter.clear()
            frames_since_last_pred = 0
            idle_frames = 0
            # print(">> START")
        continue

    # =====================
    # GESTURE state
    # =====================
    gesture_frames += 1
    frames_since_last_pred += 1

    # sample only when gesture is likely finishing (more stable)
    should_sample = (gesture_frames >= MIN_GESTURE_FRAMES) and (e_s < start_th)

    if should_sample and frames_since_last_pred >= PRED_INTERVAL_FRAMES:
        frames_since_last_pred = 0
        pred, conf = predict_from_buffer(buffer)
        pred_counter[pred] += 1
        conf_counter[pred] += conf

    # End detection with hysteresis
    if e_s < end_th:
        quiet += 1
    else:
        quiet = 0

    force_end = gesture_frames >= MAX_GESTURE_FRAMES

    if quiet >= END_QUIET_FRAMES or force_end:
        in_gesture = False
        cooldown = COOLDOWN_FRAMES  # ✅ IMPORTANT: block immediate retriggers

        # ignore tiny spikes
        if gesture_frames < MIN_GESTURE_FRAMES:
            quiet = 0
            gesture_frames = 0
            pred_counter.clear()
            conf_counter.clear()
            continue

        # If we never sampled predictions, do one final prediction
        if len(pred_counter) == 0:
            pred, conf = predict_from_buffer(buffer)
            pred_counter[pred] += 1
            conf_counter[pred] += conf

        # Majority vote + confidence tie-break
        best_pred = None
        best_votes = -1
        best_avg_conf = -1.0

        for k, votes in pred_counter.items():
            avg_conf = conf_counter[k] / max(1, votes)
            if votes > best_votes or (votes == best_votes and avg_conf > best_avg_conf):
                best_pred = k
                best_votes = votes
                best_avg_conf = avg_conf

        final_conf = float(best_avg_conf)
        total_votes = sum(pred_counter.values())
        vote_frac = best_votes / max(1, total_votes)

        # --- STEP 3: vote quality gate ---
        if total_votes < MIN_VOTES:
            print(f"❓ Gesture: UNKNOWN | too few votes ({total_votes})")
        elif vote_frac < MIN_VOTE_FRAC:
            print(
                f"❓ Gesture: UNKNOWN | weak agreement "
                f"({best_votes}/{total_votes} = {vote_frac:.2f})"
            )
        elif final_conf < CONF_TH:
            print(
                f"❓ Gesture: UNKNOWN | low confidence "
                f"(conf≈{final_conf:.3f})"
            )
        else:
            print(
                f"✅ Gesture: {id2label[best_pred]} "
                f"| conf≈{final_conf:.3f} "
                f"| votes={best_votes}/{total_votes}"
            )


        # reset gesture variables
        quiet = 0
        gesture_frames = 0
        pred_counter.clear()
        conf_counter.clear()
        frames_since_last_pred = 0
