import cv2
from edge_detector import custom_edge_detection
from contour_detector import custom_contour_detection, draw_custom_contours

# Load image
image = cv2.imread("images/sample.jpg")
if image is None:
    print("❌ No image found in 'images/' folder.")
    exit()

# Custom edge detection
edge_map = custom_edge_detection(image, threshold=50)

# Custom contour detection
contours = custom_contour_detection(edge_map)

# Draw contours
contour_img = draw_custom_contours(image, contours)

# Show results
cv2.imshow("Original Image", image)
cv2.imshow("Custom Edge Detection", edge_map)
cv2.imshow("Custom Contours", contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
