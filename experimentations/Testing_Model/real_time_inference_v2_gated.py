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

# SavedModel folder (recommended)
MODEL_DIR = "cnn_lstm_savedmodel"

# Window used by model during training
T = 512
F = 9

# --- Gating / UX ---
CONF_TH = 0.70              # minimum confidence to accept prediction
PRED_SAMPLES = 5            # run inference this many times near gesture end
PRED_INTERVAL_FRAMES = 8    # frames between predictions (8 @100Hz = 80ms)

# --- Energy-based gesture detection ---
ENERGY_SMOOTH_W = 8         # moving average window (frames)
START_TH_RATIO = 0.22       # relative threshold vs recent max (robust)
END_TH_RATIO = 0.16         # lower than start to add hysteresis
END_QUIET_FRAMES = 20       # quiet frames to mark gesture end (~0.2s at 100Hz)
MIN_GESTURE_FRAMES = 25     # ignore tiny spikes (<0.25s)
MAX_GESTURE_FRAMES = 250    # force end if too long (~2.5s) (safety)

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

# Auto-detect input/output keys from signature
input_keys = list(infer.structured_input_signature[1].keys())
output_keys = list(infer.structured_outputs.keys())

if len(input_keys) != 1 or len(output_keys) != 1:
    print("⚠️ Signature keys not single. Inputs:", input_keys, "Outputs:", output_keys)
    print("Edit the script to select correct keys.")
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
    # row9: [emg1,emg2,emg3,ax,ay,az,gx,gy,gz]
    return float(np.sum(np.abs(row9[:3])))

def moving_avg_update(buf: deque, new_val: float, w: int) -> float:
    buf.append(new_val)
    return float(sum(buf) / len(buf))

def predict_from_buffer(buffer: deque) -> tuple[int, float]:
    """
    Returns: (pred_id, confidence)
    """
    X = np.array(buffer, dtype=np.float32)  # (T,9)

    # Match training: EMG DC removal
    X[:, :3] -= X[:, :3].mean(axis=0, keepdims=True)

    # Normalize
    X = normalize(X)

    # Batch + float32
    X = X[np.newaxis, ...].astype(np.float32)  # (1,T,9)

    out = infer(**{IN_KEY: tf.constant(X)})
    probs = out[OUT_KEY].numpy()[0]  # (10,)
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return pred, conf

# =====================
# Serial setup
# =====================
ser = serial.Serial(PORT, BAUD, timeout=1)
print("Listening on", PORT)

# Rolling window for model input
buffer = deque(maxlen=T)

# Buffers for energy smoothing + adaptive thresholds
energy_raw = deque(maxlen=ENERGY_SMOOTH_W)
recent_energy_max = deque(maxlen=250)  # ~2.5s history at 100Hz

# Gesture state machine
in_gesture = False
quiet = 0
gesture_frames = 0

# For “predict near end”
pred_counter = Counter()
conf_counter = Counter()
frames_since_last_pred = 0

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if not line or "t_ms" in line:
        continue

    parts = line.split(",")
    if len(parts) != 10:
        continue

    try:
        vals = np.array(parts[1:], dtype=np.float32)  # drop timestamp -> (9,)
    except:
        continue

    buffer.append(vals)

    # Need full window for inference
    if len(buffer) < T:
        continue

    # --- Energy tracking ---
    e = emg_energy(vals)
    e_s = moving_avg_update(energy_raw, e, ENERGY_SMOOTH_W)

    recent_energy_max.append(e_s)
    emax = max(recent_energy_max) if recent_energy_max else e_s

    # Adaptive thresholds (relative to recent max)
    start_th = START_TH_RATIO * emax
    end_th = END_TH_RATIO * emax

    # --- Gesture state machine ---
    if not in_gesture:
        # Start condition
        if e_s > start_th:
            in_gesture = True
            quiet = 0
            gesture_frames = 0
            pred_counter.clear()
            conf_counter.clear()
            frames_since_last_pred = 0
            # print(">> Gesture START")
        continue

    # In gesture
    gesture_frames += 1
    frames_since_last_pred += 1

    # Predict only near the end: we start collecting predictions once signal begins dropping
    # heuristic: if energy is below start_th but still in gesture, begin sampling predictions
    should_sample_pred = (e_s < start_th) or (gesture_frames > MIN_GESTURE_FRAMES)

    if should_sample_pred and frames_since_last_pred >= PRED_INTERVAL_FRAMES:
        frames_since_last_pred = 0
        pred, conf = predict_from_buffer(buffer)
        pred_counter[pred] += 1
        # track confidence sum for tie-break
        conf_counter[pred] += conf

    # End detection
    if e_s < end_th:
        quiet += 1
    else:
        quiet = 0

    # Safety end
    force_end = gesture_frames >= MAX_GESTURE_FRAMES

    if quiet >= END_QUIET_FRAMES or force_end:
        # print("<< Gesture END")
        in_gesture = False

        if gesture_frames < MIN_GESTURE_FRAMES:
            # Too short, ignore as noise
            # print("(ignored short spike)")
            continue

        if len(pred_counter) == 0:
            # If we never sampled predictions, do one final prediction
            pred, conf = predict_from_buffer(buffer)
            pred_counter[pred] += 1
            conf_counter[pred] += conf

        # Choose majority; tie-break using avg confidence
        best_pred = None
        best_votes = -1
        best_score = -1.0

        for k, votes in pred_counter.items():
            avg_conf = conf_counter[k] / max(1, votes)
            score = votes + 0.25 * avg_conf  # votes dominates, confidence breaks ties
            if votes > best_votes or (votes == best_votes and score > best_score):
                best_pred = k
                best_votes = votes
                best_score = score

        # Estimate final confidence as avg of that class during sampling
        final_conf = float(conf_counter[best_pred] / max(1, pred_counter[best_pred]))

        if final_conf >= CONF_TH:
            print(f"✅ Gesture: {id2label[best_pred]}  | conf≈{final_conf:.3f} | votes={best_votes}")
        else:
            print(f"❓ Gesture: UNKNOWN | conf≈{final_conf:.3f} (below {CONF_TH})")

        # reset
        quiet = 0
        gesture_frames = 0
        pred_counter.clear()
        conf_counter.clear()
        frames_since_last_pred = 0
