import cv2
import numpy as np

print("✅ OpenCV and NumPy are working!")

# Load a test image
img = cv2.imread("images/sample.jpg")
if img is None:
    print("❌ No image found! Add one to the 'images/' folder.")
else:
    print("✅ Image loaded successfully!")
    cv2.imshow("Sample", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
