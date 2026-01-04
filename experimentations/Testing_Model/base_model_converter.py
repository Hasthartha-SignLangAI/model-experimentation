import tensorflow as tf

SAVEDMODEL_DIR = "cnn_lstm_with_idle_savedmodel"
OUT_TFLITE = "cnn_lstm_with_idle_fp32.tflite"

converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS   # important for LSTM safety
]

tflite_model = converter.convert()

with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_model)

print("✅ Saved", OUT_TFLITE)
