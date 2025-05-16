import tensorflow as tf
import numpy as np
from PIL import Image
import os
from utils import preprocess_image

def load_training_data():
    X_train = []
    y_train = []
    
    # Load images from both 0 and 1 folders
    for digit in [0, 1]:
        folder_path = os.path.join('train-data', str(digit))
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path)
                processed_image = preprocess_image(image)
                X_train.append(processed_image)
                y_train.append(digit)
    
    return np.array(X_train), np.array(y_train)

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def train_model():
    print("Loading training data...")
    X_train, y_train = load_training_data()
    print(f"Training data loaded: {len(X_train)} images")

    model = create_model()
    
    print("Training the model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    print("Model training completed!")
    
    return model

# Initialize and train the model
model = train_model() 