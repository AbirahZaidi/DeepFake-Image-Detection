from tensorflow.keras.models import load_model

model = load_model("model/deepfake_detector.h5")
model.summary()
