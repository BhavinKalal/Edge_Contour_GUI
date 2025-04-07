import numpy as np
import cv2

def is_valid(px, py, height, width):
    return 0 <= px < height and 0 <= py < width

def flood_fill(edge_map, visited, start_x, start_y):
    """
    Simple flood-fill to collect all connected pixels in a region.
    """
    height, width = edge_map.shape
    stack = [(start_x, start_y)]
    contour = []

    while stack:
        x, y = stack.pop()
        if not is_valid(x, y, height, width):
            continue
        if visited[x, y] or edge_map[x, y] == 0:
            continue

        visited[x, y] = True
        contour.append((x, y))

        # 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    stack.append((x + dx, y + dy))

    return contour

def custom_contour_detection(edge_map):
    """
    Detect contours by finding connected white regions in binary edge map.
    :param edge_map: Binary edge-detected image (255=white, 0=black)
    :return: List of contours (each contour is a list of (x, y) points)
    """
    height, width = edge_map.shape
    visited = np.zeros_like(edge_map, dtype=bool)
    contours = []

    for i in range(height):
        for j in range(width):
            if edge_map[i, j] == 255 and not visited[i, j]:
                contour = flood_fill(edge_map, visited, i, j)
                if len(contour) > 30:  # filter small noise
                    contours.append(contour)

    return contours

def draw_custom_contours(image, contours, color=(0, 255, 0)):
    """
    Draw contours on the original image.
    :param image: Original BGR image
    :param contours: List of contours
    :param color: Color for drawing
    :return: Image with contours drawn
    """
    contour_image = image.copy()
    for contour in contours:
        for x, y in contour:
            cv2.circle(contour_image, (y, x), 1, color, -1)
    return contour_image
