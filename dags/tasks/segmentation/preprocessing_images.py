""" Preprocessing the images"""
import numpy as np
import cv2


def preprocess_array(image, target_size=(512, 512)):
    """Preprocess an image already loaded as a numpy array (like imageio.imread)."""
    if len(image.shape) == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_resized = cv2.resize(image_bgr, target_size)
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        image_resized = cv2.resize(image, target_size)
        gray = image_resized.astype(np.float32)

    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
    return image_resized, gray
