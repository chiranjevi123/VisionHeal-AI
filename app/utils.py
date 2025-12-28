import numpy as np
import cv2

def preprocess_image(image):
    """Standardize input for MobileNetV2 (224x224)"""
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalization
    return np.expand_dims(img, axis=0)

def generate_gradcam(model, img_array, layer_name):
    """Placeholder for Grad-CAM logic to highlight disease regions"""
    # Logic to compute gradients and generate heatmap overlay
    pass