from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})   # ← THIS LINE IS THE REAL FIX

model = load_model(r"model/deepfake_detector.h5")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file).convert('RGB')
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)[0][0]
    result = "Real" if prediction >= 0.5 else "Fake"
    confidence = float(prediction if result == "Real" else 1 - prediction)

    return jsonify({'result': result, 'confidence': confidence})

if __name__ == "__main__":
    app.run(debug=True)
