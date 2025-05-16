from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
from utils import preprocess_image
from model import model

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read and preprocess the image
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        processed_image = preprocess_image(image)
        flattened_input = processed_image.flatten().reshape(1, 784)
        
        # Make prediction
        prediction = model.predict(flattened_input)
        prob = float(prediction[0][0])
        predicted_digit = int(prob > 0.5)
        confidence = prob if predicted_digit == 1 else 1 - prob
        
        return jsonify({
            'prediction': predicted_digit,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 