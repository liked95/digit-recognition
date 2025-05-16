import tensorflow as tf
import numpy as np
import os
from PIL import Image
from utils import preprocess_image

def load_data(cache_path='train_data_cache.npz'):
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path)
        X, y = data['X'], data['y']
        return X, y

    print("Processing images and caching data...")
    X = []
    y = []
    for digit in [0, 1]:
        folder_path = os.path.join('train-data', str(digit))
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path)
                processed_image = preprocess_image(image)  # shape (28, 28)
                X.append(processed_image.flatten())        # flatten to (784,)
                y.append(digit)
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"Cached data saved to {cache_path}")
    return X, y

# Load data
X, y = load_data()

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(25, activation='sigmoid'),
    tf.keras.layers.Dense(15, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

# Train the model
model.fit(X, y, epochs=10) 