"""
real_time_inference_optionA_with_fewshot_INTERACTIVE_v2.py

Option A (delta-energy start/end) + Base classifier (with idle class) + Few-shot fallback
+ Interactive terminal menu:
  [1] Real-time prediction
  [2] Add new gesture (Enroll few-shot) -> updates fewshot_db.json live

ENROLL MODE improvements:
✅ Countdown before each sample
✅ Post-sample lockout (prevents rapid multi-count from one gesture)
✅ Requires "rest/idle stability" before arming next sample

Run:
  python real_time_inference_optionA_with_fewshot_INTERACTIVE_v2.py
"""

import time, json, sys
import numpy as np
import tensorflow as tf
import serial
from collections import deque, Counter

# =====================
# CONFIG
# =====================
PORT = "/dev/tty.usbserial-0001"
BAUD = 115200

BASE_MODEL_DIR = "cnn_lstm_with_idle_savedmodel"
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
BASE_CONF_TH = 0.80
IDLE_GATE_TH = 0.60

# --- Few-shot thresholds (cosine similarity) ---
FEWSHOT_SIM_TH = 0.78
FEWSHOT_MARGIN = 0.06

# Voting in gesture
PRED_INTERVAL_FRAMES = 6
MIN_VOTES = 6
MIN_STABLE_FRAC = 0.70

# --- ENROLL MODE controls (NEW) ---
ENROLL_COUNTDOWN_SEC = 3          # 3..2..1..GO
ENROLL_LOCKOUT_SEC = 1.2          # after capturing one sample, ignore triggers for this long
ENROLL_REQUIRE_IDLE_SEC = 0.8     # must be stable/idle this long before next sample is armed
ENROLL_IDLE_MARGIN = 1.10         # considered "idle-ish" if d_s < end_th * margin


# =====================
# Terminal menu helpers
# =====================
def ask_menu_choice():
    print("\n==============================")
    print("Select mode:")
    print("  1) Real-time prediction")
    print("  2) Add new gesture (Enroll few-shot)")
    print("==============================")
    while True:
        c = input("Enter 1 or 2: ").strip()
        if c in ("1", "2"):
            return c
        print("Invalid input. Please type 1 or 2.")

def ask_enroll_details(existing_words):
    print("\n--- Enroll new gesture ---")
    while True:
        w = input("Enter new word name (no spaces): ").strip().lower()
        if not w:
            print("Word cannot be empty.")
            continue
        if " " in w:
            print("No spaces allowed. Use underscores if needed.")
            continue
        break

    if w in existing_words:
        print(f"⚠️ Word '{w}' already exists in few-shot DB.")
        ans = input("Overwrite/continue adding more samples to it? (y/n): ").strip().lower()
        if ans != "y":
            print("Cancelled enrollment.")
            return None, None

    while True:
        k = input("How many samples to record? (e.g., 10): ").strip()
        try:
            k = int(k)
            if k <= 0:
                raise ValueError
            break
        except:
            print("Enter a valid positive integer.")

    print(f"\n✅ Enrollment ready: word='{w}' samples={k}")
    return w, k

def countdown(sec: int):
    for i in range(sec, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)
    print("  GO! Perform gesture now ✋")


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
# Load SavedModel signature (auto single in/out)
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

idle_id = label2id.get("idle", None)
if idle_id is None:
    print("⚠️ No 'idle' class in label_map.json. Idle gating disabled.")


# =====================
# Few-shot DB loader/saver
# =====================
def load_fewshot_db():
    with open(FEWSHOT_DB, "r") as f:
        db = json.load(f)

    proto_dict = db["prototypes"] if (isinstance(db, dict) and "prototypes" in db) else db
    stats = db.get("stats", {}) if isinstance(db, dict) else {}

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
    return fewshot_proto, stats

fewshot_proto, fewshot_stats = load_fewshot_db()
print("Loaded few-shot prototypes:", list(fewshot_proto.keys()))
print("Prototype dims:", {k: tuple(v.shape) for k, v in fewshot_proto.items()})

def save_fewshot_db():
    out = {
        "prototypes": {k: v.tolist() for k, v in fewshot_proto.items()},
        "stats": fewshot_stats
    }
    with open(FEWSHOT_DB, "w") as f:
        json.dump(out, f, indent=2)
    print("💾 Updated:", FEWSHOT_DB)

def update_fewshot_prototype(word: str, emb: np.ndarray):
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    if word not in fewshot_proto:
        fewshot_proto[word] = emb
        fewshot_stats[word] = {"n": 1}
        return

    n = int(fewshot_stats.get(word, {}).get("n", 1))
    proto = fewshot_proto[word]

    new_proto = (proto * n + emb) / (n + 1)
    new_proto = new_proto / (np.linalg.norm(new_proto) + 1e-8)

    fewshot_proto[word] = new_proto
    fewshot_stats[word] = {"n": n + 1}


# =====================
# Signal helpers
# =====================
def emg_dc_remove(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, :3] -= X[:, :3].mean(axis=0, keepdims=True)
    return X

def emg_energy_delta(row9: np.ndarray, idle_emg_mean: np.ndarray) -> float:
    d = np.abs(row9[:3] - idle_emg_mean)
    return float(np.sum(d))

def moving_avg(buf: deque, x: float) -> float:
    buf.append(x)
    return float(sum(buf) / len(buf))

def base_predict(buffer: deque):
    X = np.array(buffer, dtype=np.float32)
    X = emg_dc_remove(X)
    X = normalize(X)
    X = X[np.newaxis, ...].astype(np.float32)

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
        vals = np.array(parts[1:], dtype=np.float32)
    except:
        continue
    idle_emg_rows.append(vals[:3])

idle_emg_rows = np.array(idle_emg_rows, dtype=np.float32)
idle_emg_mean = idle_emg_rows.mean(axis=0) if len(idle_emg_rows) else np.zeros((3,), dtype=np.float32)

deltas = np.array([float(np.sum(np.abs(r - idle_emg_mean))) for r in idle_emg_rows], dtype=np.float32)
median = float(np.median(deltas)) if len(deltas) else 0.0
mad = float(np.median(np.abs(deltas - median)) + 1e-6) if len(deltas) else 1.0
robust_std = 1.4826 * mad

start_th = median + K_START * robust_std
end_th   = median + K_END   * robust_std

print(f"Idle EMG mean: {idle_emg_mean.tolist()} (raw)")
print(f"Idle delta-energy: median={median:.2f}, robust_std={robust_std:.2f}")
print(f"start_th={start_th:.2f}  end_th={end_th:.2f}")


# =====================
# Choose mode
# =====================
choice = ask_menu_choice()

ENROLLING = False
ENROLL_WORD = None
ENROLL_K = 0
enroll_count = 0

# Enrollment arming state
enroll_armed = False
enroll_lockout_until = 0.0
idle_stable_start = None

if choice == "2":
    ENROLL_WORD, ENROLL_K = ask_enroll_details(existing_words=set(fewshot_proto.keys()))
    if ENROLL_WORD is None:
        print("Exiting.")
        sys.exit(0)
    ENROLLING = True
    enroll_count = 0

    print("\nENROLL MODE")
    print("Rules:")
    print(" - Script will wait until your arm is relaxed (idle stable)")
    print(" - Then it will do a countdown for each sample")
    print(" - After each capture, it will lockout briefly to avoid double counting\n")
else:
    print("\nPREDICTION MODE. Ctrl+C to stop.\n")


# =====================
# Main loop
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

print("\nRunning... Ctrl+C to stop.\n")

while True:
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

    buffer.append(vals)
    if len(buffer) < T:
        continue

    d = emg_energy_delta(vals, idle_emg_mean)
    d_s = moving_avg(energy_buf, d)

    now = time.time()

    # ---- ENROLL arming logic (only when not in_gesture) ----
    if ENROLLING and (not in_gesture):
        # lockout after a capture: don't arm immediately
        if now < enroll_lockout_until:
            continue

        # require stable idle before each sample
        idle_like = (d_s < (end_th * ENROLL_IDLE_MARGIN))
        if idle_like:
            if idle_stable_start is None:
                idle_stable_start = now
            elif (now - idle_stable_start) >= ENROLL_REQUIRE_IDLE_SEC and not enroll_armed:
                enroll_armed = True
                print(f"\nSample {enroll_count+1}/{ENROLL_K} for '{ENROLL_WORD}'")
                print("Relax... countdown starting:")
                countdown(ENROLL_COUNTDOWN_SEC)
        else:
            idle_stable_start = None

    # cooldown countdown (prediction state machine)
    if cooldown > 0:
        cooldown -= 1

    # -------------------- IDLE STATE --------------------
    if not in_gesture:
        # In enroll mode: only allow gesture start if armed
        allow_start = True
        if ENROLLING:
            allow_start = enroll_armed and (now >= enroll_lockout_until)

        if allow_start and cooldown == 0 and d_s > start_th:
            in_gesture = True
            quiet = 0
            frames = 0

            pred_counter.clear()
            conf_sum.clear()
            idle_sum = 0.0
            votes_total = 0
            frames_since_pred = 0
        continue

    # -------------------- GESTURE STATE --------------------
    frames += 1
    frames_since_pred += 1

    if d_s < end_th:
        quiet += 1
    else:
        quiet = 0

    end_dyn = (quiet >= END_QUIET_FRAMES)
    force_end = (frames >= MAX_GESTURE_FRAMES)

    # collect base votes only for prediction mode
    if (not ENROLLING) and frames >= MIN_GESTURE_FRAMES and frames_since_pred >= PRED_INTERVAL_FRAMES:
        frames_since_pred = 0
        pred, conf, idle_p, _ = base_predict(buffer)
        pred_counter[pred] += 1
        conf_sum[pred] += conf
        idle_sum += idle_p
        votes_total += 1

    if not (end_dyn or force_end):
        continue

    # finalize gesture
    in_gesture = False
    cooldown = COOLDOWN_FRAMES
    end_reason = "end_dyn" if end_dyn else "force_end"

    if frames < MIN_GESTURE_FRAMES:
        # ignore tiny spikes
        quiet = 0
        frames = 0
        pred_counter.clear()
        conf_sum.clear()
        idle_sum = 0.0
        votes_total = 0
        frames_since_pred = 0
        continue

    # =====================
    # ENROLLMENT PATH
    # =====================
    if ENROLLING:
        emb = encode_embed(buffer)
        update_fewshot_prototype(ENROLL_WORD, emb)
        enroll_count += 1
        print(f"🧩 ENROLLED '{ENROLL_WORD}' {enroll_count}/{ENROLL_K} | end={end_reason}")

        # after capture: lockout + require idle again
        enroll_lockout_until = time.time() + ENROLL_LOCKOUT_SEC
        enroll_armed = False
        idle_stable_start = None

        if enroll_count >= ENROLL_K:
            save_fewshot_db()
            print(f"\n✅ Enrollment complete for '{ENROLL_WORD}'.")
            print("Now run the script again and choose Prediction mode.\n")
            sys.exit(0)

        # reset per-gesture vars
        quiet = 0
        frames = 0
        pred_counter.clear()
        conf_sum.clear()
        idle_sum = 0.0
        votes_total = 0
        frames_since_pred = 0
        continue

    # =====================
    # PREDICTION PATH
    # =====================
    if votes_total == 0:
        pred, conf, idle_p, _ = base_predict(buffer)
        pred_counter[pred] += 1
        conf_sum[pred] += conf
        idle_sum += idle_p
        votes_total = 1

    best = pred_counter.most_common(1)[0][0]
    best_votes = pred_counter[best]
    stable = best_votes / max(1, votes_total)
    base_conf = float(conf_sum[best] / max(1, best_votes))
    idleP = float(idle_sum / max(1, votes_total))

    accepted_base = (
        votes_total >= MIN_VOTES and
        stable >= MIN_STABLE_FRAC and
        base_conf >= BASE_CONF_TH and
        not (idle_id is not None and idleP >= IDLE_GATE_TH) and
        (idle_id is None or best != idle_id)
    )

    if accepted_base:
        label = id2label[best]
        print(f"✅ {label} | conf={base_conf:.3f} | stable={stable:.2f} | frames={frames} | end={end_reason}")
    else:
        emb = encode_embed(buffer)
        top1, s1, margin, _, _ = fewshot_predict(emb)
        very_idle = (idle_id is not None and idleP >= 0.90)

        if (not very_idle) and (s1 >= FEWSHOT_SIM_TH) and (margin >= FEWSHOT_MARGIN):
            print(f"🧩 FEWSHOT {top1} | sim={s1:.3f} | margin={margin:.3f} | frames={frames} | end={end_reason}")
        else:
            print(f"❓ UNKNOWN | base_conf={base_conf:.3f} stable={stable:.2f} idleP={idleP:.2f} "
                  f"| fewshot_top={top1}:{s1:.3f} m={margin:.3f} | frames={frames} | end={end_reason}")

    # reset per-gesture
    quiet = 0
    frames = 0
    pred_counter.clear()
    conf_sum.clear()
    idle_sum = 0.0
    votes_total = 0
    frames_since_pred = 0
