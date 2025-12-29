from collections import deque
import numpy as np
import serial
import tensorflow as tf
import json

PORT = "/dev/tty.usbserial-0001"
BAUD = 115200
T = 512
F = 9

# Load model
loaded = tf.saved_model.load("cnn_lstm_savedmodel")
infer = loaded.signatures["serving_default"]


# Load label map
with open("label_map.json") as f:
    label_map = json.load(f)
id2label = {v: k for k, v in label_map.items()}

# Load scaler
with open("scaler_params.json") as f:
    sc = json.load(f)
mean = np.array(sc["mean"])
scale = np.array(sc["scale"])

ser = serial.Serial(PORT, BAUD, timeout=1)
print("Listening on", PORT)

buffer = deque(maxlen=T)

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if not line:
        continue

    parts = line.split(",")
    if len(parts) != 10:
        continue

    try:
        vals = np.array(parts[1:], dtype=np.float32)  # drop timestamp
    except:
        continue

    buffer.append(vals)

    # 🔥 Force prediction once buffer is full
    if len(buffer) == T:
        X = np.array(buffer)

        # EMG DC removal
        X[:, :3] -= X[:, :3].mean(axis=0, keepdims=True)

        # Normalize
        X = (X - mean) / scale

        # Batch dimension + dtype
        X = X[np.newaxis, ...].astype(np.float32)

        # Inference
        out = infer(keras_tensor_73=tf.constant(X))
        probs = out["output_0"].numpy()[0]

        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        print(f"Predicted: {id2label[pred]} | Confidence: {conf:.3f}")

