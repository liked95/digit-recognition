import numpy as np
from PIL import Image


def preprocess_image(image):
    # Extract alpha channel if present, otherwise create a full opaque alpha
    if image.mode == 'RGBA':
        alpha = image.split()[-1]
    else:
        # If no alpha, create a fully opaque alpha channel
        alpha = Image.new('L', image.size, 255)

    # Resize to 28x28
    alpha = alpha.resize((28, 28))
    alpha_np = np.array(alpha).astype('float32')

    # Z-normalize
    mean = alpha_np.mean()
    std = alpha_np.std() if alpha_np.std() > 0 else 1.0
    alpha_np = (alpha_np - mean) / std

    return alpha_np
