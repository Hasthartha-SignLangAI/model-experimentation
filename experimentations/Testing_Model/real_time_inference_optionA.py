import time
import json
import numpy as np
import tensorflow as tf
import serial
from collections import deque, Counter

# =====================
# CONFIG
# =====================
PORT = "/dev/tty.usbserial-0001"   # change if needed
BAUD = 115200

MODEL_DIR = "cnn_lstm_with_idle_savedmodel"  # folder you extracted from zip

T = 512
F = 9

# Inference cadence (your ESP looks ~100Hz)
PRED_EVERY_N_FRAMES = 4     # predict every 4 frames (~25 predictions/sec)

# Smoothing / stability
VOTE_WIN = 25              # how many recent predictions we vote over (~1s if 25 preds/sec)
MIN_STABLE_FRAC = 0.70     # winner must get >=70% of votes
MIN_AVG_CONF = 0.80        # winner avg confidence must be >=0.80

# State gating
IDLE_LABEL_NAME = "idle"   # must match your folder/class name exactly
REQUIRE_IDLE_BEFORE_NEW = True
IDLE_STABLE_FRAC = 0.75
IDLE_MIN_CONF = 0.70

COOLDOWN_SEC = 0.60        # after emitting a word, ignore new words briefly

# =====================
# Load label map + scaler
# =====================
with open("label_map.json") as f:
    label2id = json.load(f)
id2label = {int(v): k for k, v in label2id.items()}  # ensure int keys

if IDLE_LABEL_NAME not in label2id:
    raise RuntimeError(f"'{IDLE_LABEL_NAME}' not found in label_map.json. Found: {list(label2id.keys())}")

IDLE_ID = int(label2id[IDLE_LABEL_NAME])

with open("scaler_params.json") as f:
    sc = json.load(f)
mean = np.array(sc["mean"], dtype=np.float32)
scale = np.array(sc["scale"], dtype=np.float32)

def normalize(X):
    return (X - mean) / (scale + 1e-6)

# =====================
# Load SavedModel signature
# =====================
loaded = tf.saved_model.load(MODEL_DIR)
infer = loaded.signatures["serving_default"]

input_keys = list(infer.structured_input_signature[1].keys())
output_keys = list(infer.structured_outputs.keys())

if len(input_keys) != 1 or len(output_keys) != 1:
    raise RuntimeError(f"Signature not single I/O. Inputs={input_keys}, Outputs={output_keys}")

IN_KEY = input_keys[0]
OUT_KEY = output_keys[0]

print("Using SavedModel signature:")
print(" IN_KEY :", IN_KEY)
print(" OUT_KEY:", OUT_KEY)

# =====================
# Prediction from rolling buffer
# =====================
def predict_from_buffer(buf):
    X = np.array(buf, dtype=np.float32)            # (T,9)
    # match training: EMG DC remove
    X[:, :3] -= X[:, :3].mean(axis=0, keepdims=True)
    # normalize (same scaler)
    X = normalize(X)
    # batch float32
    X = X[np.newaxis, ...].astype(np.float32)      # (1,T,9)

    out = infer(**{IN_KEY: tf.constant(X)})
    probs = out[OUT_KEY].numpy()[0]                # (C,)
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return pred, conf, probs

# =====================
# Serial
# =====================
ser = serial.Serial(PORT, BAUD, timeout=1)
print("Listening on", PORT)

# Rolling raw window
buffer = deque(maxlen=T)

# Recent predictions for voting
pred_hist = deque(maxlen=VOTE_WIN)   # stores (pred_id, conf)

frame_count = 0
last_emitted_time = 0.0
idle_ready = False  # becomes True when we see stable idle

def vote_summary(hist):
    # returns (best_id, frac, avg_conf_best, avg_conf_idle, idle_frac)
    if len(hist) == 0:
        return None, 0.0, 0.0, 0.0, 0.0

    preds = [p for (p, c) in hist]
    confs = [c for (p, c) in hist]

    cnt = Counter(preds)
    best_id, best_votes = cnt.most_common(1)[0]
    frac = best_votes / len(hist)

    # avg conf for best class
    best_confs = [c for (p, c) in hist if p == best_id]
    avg_conf_best = float(np.mean(best_confs)) if best_confs else 0.0

    idle_votes = cnt.get(IDLE_ID, 0)
    idle_frac = idle_votes / len(hist)
    idle_confs = [c for (p, c) in hist if p == IDLE_ID]
    avg_conf_idle = float(np.mean(idle_confs)) if idle_confs else 0.0

    return best_id, frac, avg_conf_best, avg_conf_idle, idle_frac

print("Running Option A (idle-class gating). Press Ctrl+C to stop.\n")

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if not line or "t_ms" in line:
        continue

    parts = line.split(",")
    if len(parts) != 10:
        continue

    try:
        vals = np.array(parts[1:], dtype=np.float32)  # drop timestamp => 9 features (EMG3 + IMU6)
    except:
        continue

    buffer.append(vals)
    if len(buffer) < T:
        continue

    frame_count += 1
    if frame_count % PRED_EVERY_N_FRAMES != 0:
        continue

    pred, conf, _ = predict_from_buffer(buffer)
    pred_hist.append((pred, conf))

    best_id, frac, avg_conf_best, avg_conf_idle, idle_frac = vote_summary(pred_hist)

    # Decide if we are "idle-ready"
    if idle_frac >= IDLE_STABLE_FRAC and avg_conf_idle >= IDLE_MIN_CONF:
        idle_ready = True

    now = time.time()
    in_cooldown = (now - last_emitted_time) < COOLDOWN_SEC

    # If we require idle before new, block words until idle_ready becomes True
    if REQUIRE_IDLE_BEFORE_NEW and not idle_ready:
        # (optional) you can print once in a while
        continue

    # If stable IDLE, just keep waiting (don’t emit words)
    if best_id == IDLE_ID and frac >= IDLE_STABLE_FRAC and avg_conf_best >= IDLE_MIN_CONF:
        # once we confirm idle, keep idle_ready True
        idle_ready = True
        continue

    # Candidate word (non-idle)
    if best_id is not None and best_id != IDLE_ID:
        if in_cooldown:
            continue

        if frac >= MIN_STABLE_FRAC and avg_conf_best >= MIN_AVG_CONF:
            word = id2label.get(best_id, f"class_{best_id}")
            print(f"✅ {word} | conf={avg_conf_best:.3f} | stable={frac:.2f} | win={len(pred_hist)}")
            last_emitted_time = now

            # require returning to idle before next output
            idle_ready = False
            pred_hist.clear()
