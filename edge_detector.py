import cv2
import numpy as np

def custom_edge_detection(image, threshold=50):
    """
    Custom edge detection using Sobel filters and manual thresholding.
    :param image: Input image (BGR)
    :param threshold: Threshold for edge detection (default: 50)
    :return: Binary edge map
    """
    # Step 1: Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # Step 3: Compute gradients using Sobel filters
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Step 4: Compute gradient magnitude
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    # Step 5: Apply manual threshold
    _, edge_map = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)

    return edge_map
