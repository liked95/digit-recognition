from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array and normalize
    image = np.array(image)
    image = image.astype('float32') / 255.0
    # Invert colors (MNIST format)
    image = 1 - image
    return image

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