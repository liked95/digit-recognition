import numpy as np
from PIL import Image

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