"""
real_time_inference_optionA_with_fewshot_FIXED.py

Option A (delta-energy start/end) + Base classifier (with idle class) + Few-shot fallback.
✅ Fixes fewshot_db.json structure: expects {"prototypes": {...}, "stats": {...}} like yours.
✅ Handles both single-prototype-per-word and list-of-prototypes-per-word.
✅ Robust serial parsing, stable vote logic, idle gating, cooldown, etc.

Run:
  python real_time_inference_optionA_with_fewshot_FIXED.py
"""

import time, json
import numpy as np
import tensorflow as tf
import serial
from collections import deque, Counter

# =====================
# CONFIG
# =====================
PORT = "/dev/tty.usbserial-0001"
BAUD = 115200

# Base classifier (with idle class)
BASE_MODEL_DIR = "cnn_lstm_with_idle_savedmodel"

# Metric encoder (for few-shot)
ENCODER_DIR = "./few_shot_triplet_siamese_without_gpu/encoder_savedmodel_cpu"
FEWSHOT_DB  = "./few_shot_triplet_siamese_without_gpu/fewshot_db.json"

LABEL_MAP_PATH = "label_map.json"
SCALER_PATH    = "scaler_params.json"

T = 512
F = 9

# --- Option A gating (delta-energy start/end) ---
CALIB_SECONDS = 4.0
K_START = 3.0
K_END   = 2.0

ENERGY_SMOOTH_W = 8
END_QUIET_FRAMES = 20
MIN_GESTURE_FRAMES = 35
MAX_GESTURE_FRAMES = 260
COOLDOWN_FRAMES = 40

# --- Decision (base model) ---
BASE_CONF_TH = 0.80          # accept base class if >= this (and not idle)
IDLE_GATE_TH = 0.60          # if avg idle prob >= this -> treat as idle-ish and reject base

# --- Few-shot thresholds (cosine similarity) ---
FEWSHOT_SIM_TH = 0.78        # accept if top similarity >= this
FEWSHOT_MARGIN = 0.06        # top1 - top2 must be >= this

# Voting in gesture
PRED_INTERVAL_FRAMES = 6
MIN_VOTES = 6
MIN_STABLE_FRAC = 0.70       # winner votes / total votes

# Safety / debug toggles
PRINT_REASONS_ON_ACCEPT = False


# =====================
# Load label map + scaler
# =====================
with open(LABEL_MAP_PATH, "r") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

with open(SCALER_PATH, "r") as f:
    sc = json.load(f)
mean = np.array(sc["mean"], dtype=np.float32)
scale = np.array(sc["scale"], dtype=np.float32)

def normalize(X: np.ndarray) -> np.ndarray:
    return (X - mean) / (scale + 1e-6)


# =====================
# Load SavedModel signature (auto-detect single in/out)
# =====================
def load_signature(model_dir: str):
    loaded = tf.saved_model.load(model_dir)
    infer = loaded.signatures["serving_default"]
    in_keys = list(infer.structured_input_signature[1].keys())
    out_keys = list(infer.structured_outputs.keys())
    if len(in_keys) != 1 or len(out_keys) != 1:
        raise RuntimeError(f"Signature not single IO. in={in_keys} out={out_keys}")
    return infer, in_keys[0], out_keys[0]

base_infer, BASE_IN_KEY, BASE_OUT_KEY = load_signature(BASE_MODEL_DIR)
enc_infer,  ENC_IN_KEY,  ENC_OUT_KEY  = load_signature(ENCODER_DIR)

print("Base SavedModel signature:")
print("  IN_KEY :", BASE_IN_KEY)
print("  OUT_KEY:", BASE_OUT_KEY)

print("Embed SavedModel signature:")
print("  IN_KEY :", ENC_IN_KEY)
print("  OUT_KEY:", ENC_OUT_KEY)

# Find idle index
idle_id = label2id.get("idle", None)
if idle_id is None:
    print("⚠️ No 'idle' class in label_map.json. Idle gating will be disabled.")


# =====================
# Few-shot DB (FIXED for your JSON format)
# =====================
with open(FEWSHOT_DB, "r") as f:
    few_db = json.load(f)

# Your file is:
# {
#   "prototypes": { "amma": [...128...], "nil": [...128...] },
#   "stats": {...}
# }
# But we also support older shapes for safety.
if isinstance(few_db, dict) and "prototypes" in few_db:
    proto_dict = few_db["prototypes"]
else:
    proto_dict = few_db  # fallback: assume already dict(label -> vector OR list-of-vectors)

fewshot_proto = {}
for label, v in proto_dict.items():
    arr = np.array(v, dtype=np.float32)
    if arr.ndim == 1:
        proto = arr
    elif arr.ndim == 2:
        proto = arr.mean(axis=0)
    else:
        raise ValueError(f"Unsupported prototype shape for '{label}': {arr.shape}")
    proto = proto / (np.linalg.norm(proto) + 1e-8)
    fewshot_proto[label] = proto

print("Loaded few-shot prototypes:", list(fewshot_proto.keys()))
print("Prototype dims:", {k: tuple(v.shape) for k, v in fewshot_proto.items()})


# =====================
# Helpers
# =====================
def emg_dc_remove(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, :3] -= X[:, :3].mean(axis=0, keepdims=True)
    return X

def emg_energy_delta(row9: np.ndarray, idle_emg_mean: np.ndarray) -> float:
    # row9: [emg1 emg2 emg3 ax ay az gx gy gz]
    d = np.abs(row9[:3] - idle_emg_mean)
    return float(np.sum(d))

def moving_avg(buf: deque, x: float) -> float:
    buf.append(x)
    return float(sum(buf) / len(buf))

def base_predict(buffer: deque):
    X = np.array(buffer, dtype=np.float32)      # (T,9)
    X = emg_dc_remove(X)
    X = normalize(X)
    X = X[np.newaxis, ...].astype(np.float32)   # (1,T,9)

    out = base_infer(**{BASE_IN_KEY: tf.constant(X)})
    probs = out[BASE_OUT_KEY].numpy()[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    idle_p = float(probs[idle_id]) if idle_id is not None else 0.0
    return pred, conf, idle_p, probs

def encode_embed(buffer: deque):
    X = np.array(buffer, dtype=np.float32)
    X = emg_dc_remove(X)
    X = normalize(X)
    X = X[np.newaxis, ...].astype(np.float32)

    out = enc_infer(**{ENC_IN_KEY: tf.constant(X)})
    emb = out[ENC_OUT_KEY].numpy()[0].astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb

def fewshot_predict(emb: np.ndarray):
    # cosine similarity since vectors are normalized
    scores = {k: float(np.dot(emb, p)) for k, p in fewshot_proto.items()}
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top1, s1 = items[0]
    top2, s2 = items[1] if len(items) > 1 else ("_", -1.0)
    margin = s1 - s2
    return top1, s1, margin, top2, s2


# =====================
# Serial + idle calibration
# =====================
ser = serial.Serial(PORT, BAUD, timeout=1)
print("Listening on", PORT)
print(f"Calibrating idle for {CALIB_SECONDS}s (relax arm)...")

t0 = time.time()
idle_emg_rows = []

while time.time() - t0 < CALIB_SECONDS:
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

    idle_emg_rows.append(vals[:3])

idle_emg_rows = np.array(idle_emg_rows, dtype=np.float32)
idle_emg_mean = idle_emg_rows.mean(axis=0) if len(idle_emg_rows) else np.zeros((3,), dtype=np.float32)

# Build delta-energy baseline distribution during calibration
deltas = []
for r in idle_emg_rows:
    deltas.append(float(np.sum(np.abs(r - idle_emg_mean))))
deltas = np.array(deltas, dtype=np.float32)

median = float(np.median(deltas)) if len(deltas) else 0.0
mad = float(np.median(np.abs(deltas - median)) + 1e-6) if len(deltas) else 1.0
robust_std = 1.4826 * mad

start_th = median + K_START * robust_std
end_th   = median + K_END   * robust_std

print(f"Idle EMG mean: {idle_emg_mean.tolist()} (raw)")
print(f"Idle delta-energy: median={median:.2f}, robust_std={robust_std:.2f}")
print(f"start_th={start_th:.2f}  end_th={end_th:.2f}")


# =====================
# Main loop (Option A + few-shot fallback)
# =====================
buffer = deque(maxlen=T)
energy_buf = deque(maxlen=ENERGY_SMOOTH_W)

in_gesture = False
quiet = 0
frames = 0
cooldown = 0

pred_counter = Counter()
conf_sum = Counter()
idle_sum = 0.0
votes_total = 0
frames_since_pred = 0
peak_delta = 0.0

print("\nRunning Option A + Few-shot fallback. Ctrl+C to stop.\n")

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if not line or "t_ms" in line:
        continue

    parts = line.split(",")
    if len(parts) != 10:
        continue

    try:
        vals = np.array(parts[1:], dtype=np.float32)  # (9,)
    except:
        continue

    buffer.append(vals)
    if len(buffer) < T:
        continue

    # delta-energy + smooth
    d = emg_energy_delta(vals, idle_emg_mean)
    d_s = moving_avg(energy_buf, d)

    if cooldown > 0:
        cooldown -= 1

    # --------------------
    # IDLE STATE
    # --------------------
    if not in_gesture:
        if cooldown == 0 and d_s > start_th:
            in_gesture = True
            quiet = 0
            frames = 0
            peak_delta = 0.0

            pred_counter.clear()
            conf_sum.clear()
            idle_sum = 0.0
            votes_total = 0
            frames_since_pred = 0
        continue

    # --------------------
    # GESTURE STATE
    # --------------------
    frames += 1
    frames_since_pred += 1
    if d_s > peak_delta:
        peak_delta = d_s

    # end detection
    if d_s < end_th:
        quiet += 1
    else:
        quiet = 0

    end_dyn = (quiet >= END_QUIET_FRAMES)
    force_end = (frames >= MAX_GESTURE_FRAMES)

    # collect votes during gesture (after a little time)
    if frames >= MIN_GESTURE_FRAMES and frames_since_pred >= PRED_INTERVAL_FRAMES:
        frames_since_pred = 0
        pred, conf, idle_p, _ = base_predict(buffer)
        pred_counter[pred] += 1
        conf_sum[pred] += conf
        idle_sum += idle_p
        votes_total += 1

    if not (end_dyn or force_end):
        continue

    # finalize this gesture
    in_gesture = False
    cooldown = COOLDOWN_FRAMES
    end_reason = "end_dyn" if end_dyn else "force_end"

    if frames < MIN_GESTURE_FRAMES:
        # too short -> ignore
        quiet = 0
        frames = 0
        peak_delta = 0.0
        pred_counter.clear()
        conf_sum.clear()
        idle_sum = 0.0
        votes_total = 0
        frames_since_pred = 0
        continue

    # if no votes collected, do one vote now
    if votes_total == 0:
        pred, conf, idle_p, _ = base_predict(buffer)
        pred_counter[pred] += 1
        conf_sum[pred] += conf
        idle_sum += idle_p
        votes_total = 1

    # pick winner
    best = pred_counter.most_common(1)[0][0]
    best_votes = pred_counter[best]
    stable = best_votes / max(1, votes_total)
    base_conf = float(conf_sum[best] / max(1, best_votes))
    idleP = float(idle_sum / max(1, votes_total))

    reasons = []
    if votes_total < MIN_VOTES:
        reasons.append("low_votes")
    if stable < MIN_STABLE_FRAC:
        reasons.append("unstable")
    if idle_id is not None and idleP >= IDLE_GATE_TH:
        reasons.append("idle_gate")
    if base_conf < BASE_CONF_TH:
        reasons.append("low_conf")

    # decide: base vs fewshot vs unknown
    accepted_base = (
        votes_total >= MIN_VOTES and
        stable >= MIN_STABLE_FRAC and
        base_conf >= BASE_CONF_TH and
        not (idle_id is not None and idleP >= IDLE_GATE_TH) and
        (idle_id is None or best != idle_id)
    )

    if accepted_base:
        label = id2label[best]
        if PRINT_REASONS_ON_ACCEPT:
            print(f"✅ {label} | conf={base_conf:.3f} | stable={stable:.2f} | votes={votes_total} idleP={idleP:.2f} | frames={frames} | end={end_reason}")
        else:
            print(f"✅ {label} | conf={base_conf:.3f} | stable={stable:.2f} | frames={frames} | end={end_reason}")
    else:
        # few-shot fallback
        emb = encode_embed(buffer)
        top1, s1, margin, top2, s2 = fewshot_predict(emb)

        # extra safety: if base looks extremely idle, don't accept few-shot
        very_idle = (idle_id is not None and idleP >= 0.90)

        if (not very_idle) and (s1 >= FEWSHOT_SIM_TH) and (margin >= FEWSHOT_MARGIN):
            print(f"🧩 FEWSHOT {top1} | sim={s1:.3f} | margin={margin:.3f} | frames={frames} | end={end_reason}")
        else:
            why = "+".join(reasons) if reasons else "rejected"
            gate_note = " | idle_block" if very_idle else ""
            print(f"❓ UNKNOWN | base_conf={base_conf:.3f} stable={stable:.2f} idleP={idleP:.2f} votes={votes_total} "
                  f"| fewshot_top={top1}:{s1:.3f} m={margin:.3f}{gate_note} | {why} | frames={frames} | end={end_reason}")

    # reset per-gesture
    quiet = 0
    frames = 0
    peak_delta = 0.0
    pred_counter.clear()
    conf_sum.clear()
    idle_sum = 0.0
    votes_total = 0
    frames_since_pred = 0
