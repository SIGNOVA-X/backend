from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS 
import tensorflow as tf
import cv2
import io

app = Flask(__name__)
CORS(app)  
# Load the correct model
model = tf.keras.models.load_model("cnn8grps_rad1_model.h5")

# Update labels based on your model output
class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # A-Z

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file found in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read and decode image
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Preprocess image to match model input
        img_resized = cv2.resize(img, (400, 400))  # as per your model input shape
        img_array = img_resized.reshape(1, 400, 400, 3)
        img_array = img_array / 255.0  # normalize if your model was trained this way

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
