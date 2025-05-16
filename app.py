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
        
        # Make prediction
        prediction = model.predict(np.expand_dims(processed_image, axis=0))
        predicted_digit = np.argmax(prediction[0])
        
        return jsonify({
            'prediction': int(predicted_digit),
            'confidence': float(prediction[0][predicted_digit])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 