import os, glob, json
import numpy as np
import tensorflow as tf

# =====================
# PATHS (from your screenshot)
# =====================
ENCODER_DIR  = "./few_shot_triplet_siamese_without_gpu/encoder_savedmodel_cpu"
SCALER_PATH  = "scaler_params.json"
FEWSHOT_ROOT = "./FewShot"
OUT_DB       = "./few_shot_triplet_siamese_without_gpu/fewshot_db.json"

T = 512
F = 9

# =====================
# Load scaler params
# =====================
with open(SCALER_PATH) as f:
    sc = json.load(f)
mean  = np.array(sc["mean"], dtype=np.float32)
scale = np.array(sc["scale"], dtype=np.float32)

def normalize(X):
    return (X - mean) / (scale + 1e-6)

# =====================
# Same robust loader + crop used in training
# =====================
def load_csv_robust(fp, expected_cols=10):
    with open(fp, "rb") as f:
        raw = f.read().replace(b"\x00", b"")
    text = raw.decode("utf-8", errors="ignore")

    good_rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        while line.endswith(","):
            line = line[:-1].strip()
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != expected_cols:
            continue
        if any(p == "" for p in parts):
            continue
        try:
            good_rows.append([float(p) for p in parts])
        except:
            continue

    if not good_rows:
        raise ValueError(f"No valid numeric rows in {fp}")
    return np.array(good_rows, dtype=np.float32)

def moving_average(x, w=25):
    w = max(1, int(w))
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, kernel, mode="same")

def fix_length_center(X, target_len):
    if len(X) >= target_len:
        start = (len(X) - target_len) // 2
        return X[start:start + target_len]
    pad = np.zeros((target_len - len(X), X.shape[1]), dtype=X.dtype)
    return np.vstack([X, pad])

def emg_dc_remove(X):
    X = X.copy()
    X[:, :3] -= X[:, :3].mean(axis=0, keepdims=True)
    return X

def crop_active_region_emg(X, target_len=512, smooth_w=25, thresh_ratio=0.25):
    Traw = X.shape[0]
    if Traw == 0:
        return np.zeros((target_len, X.shape[1]), dtype=np.float32)

    energy = np.sum(np.abs(X[:, :3]), axis=1)
    energy_s = moving_average(energy, w=smooth_w)

    mx = float(np.max(energy_s))
    if mx <= 1e-6:
        return fix_length_center(X, target_len)

    thresh = thresh_ratio * mx
    active = np.where(energy_s >= thresh)[0]
    if len(active) < 5:
        return fix_length_center(X, target_len)

    start = int(active[0])
    end   = int(active[-1])
    center = (start + end) // 2

    half = target_len // 2
    win_start = max(0, center - half)
    win_end = win_start + target_len
    if win_end > Traw:
        win_end = Traw
        win_start = max(0, win_end - target_len)

    cropped = X[win_start:win_end]
    if cropped.shape[0] < target_len:
        pad = np.zeros((target_len - cropped.shape[0], X.shape[1]), dtype=cropped.dtype)
        cropped = np.vstack([cropped, pad])
    return cropped

def load_one_sample(fp):
    arr = load_csv_robust(fp, expected_cols=10)  # (Traw,10)
    X = arr[:, 1:]                               # drop timestamp -> (Traw,9)
    X = emg_dc_remove(X)
    X = crop_active_region_emg(X, target_len=T)
    X = normalize(X)
    return X.astype(np.float32)

# =====================
# Load encoder SavedModel signature
# =====================
loaded = tf.saved_model.load(ENCODER_DIR)
infer = loaded.signatures["serving_default"]

IN_KEY  = list(infer.structured_input_signature[1].keys())[0]
OUT_KEY = list(infer.structured_outputs.keys())[0]

print("Encoder signature:")
print("  IN_KEY :", IN_KEY)
print("  OUT_KEY:", OUT_KEY)

def embed(X):  # X: (T,9)
    Xb = X[np.newaxis, ...].astype(np.float32)
    out = infer(**{IN_KEY: tf.constant(Xb)})
    emb = out[OUT_KEY].numpy()[0]
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb.astype(np.float32)

# =====================
# Build prototypes
# =====================
proto = {}
stats = {}

words = sorted([d for d in os.listdir(FEWSHOT_ROOT) if os.path.isdir(os.path.join(FEWSHOT_ROOT, d))])
print("Found few-shot word folders:", words)

for w in words:
    files = sorted(glob.glob(os.path.join(FEWSHOT_ROOT, w, "*.txt")))
    if not files:
        print("⚠️ No .txt files in", w)
        continue

    embs = []
    for fp in files:
        try:
            X = load_one_sample(fp)
            embs.append(embed(X))
        except Exception as e:
            print("[SKIP]", fp, "->", e)

    if not embs:
        print("⚠️ No valid embeddings for", w)
        continue

    E = np.stack(embs, axis=0)
    p = E.mean(axis=0)
    p = p / (np.linalg.norm(p) + 1e-9)

    proto[w] = p.tolist()
    stats[w] = {"n": int(len(embs))}

print("✅ Prototypes built:", stats)

with open(OUT_DB, "w") as f:
    json.dump({"prototypes": proto, "stats": stats}, f, indent=2)

print("Saved:", OUT_DB)
