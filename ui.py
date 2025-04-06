import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from edge_detector import custom_edge_detection
from contour_detector import custom_contour_detection, draw_custom_contours

class EdgeContourApp:
    def __init__(self, root):

        self.root = root

        # 🌙 Dark Mode Theme Colors
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.btn_color = "#333333"
        self.highlight_color = "#00c3ff"

        # Apply dark background
        self.root.configure(bg=self.bg_color)
        self.root.title("Custom Edge & Contour Detection")
        self.root.geometry("1000x600")
        self.image = None
        self.processed_img = None

        # Upload Button
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image,
    bg=self.btn_color, fg=self.fg_color, activebackground=self.highlight_color)
        self.upload_btn.pack(pady=5)

        self.webcam_btn = tk.Button(root, text="Capture from Webcam", command=self.capture_from_webcam,
    bg=self.btn_color, fg=self.fg_color, activebackground=self.highlight_color)
        self.webcam_btn.pack(pady=5)


        # Threshold slider
        self.threshold_val = tk.IntVar(value=50)
        self.slider = tk.Scale(root, from_=0, to=255, orient="horizontal", label="Edge Threshold", variable=self.threshold_val, bg=self.bg_color, fg=self.fg_color,
    highlightbackground=self.bg_color, troughcolor="#444444")
        self.slider.pack()

        # Live Edge Button
        self.live_edge_btn = tk.Button(root, text="Live Edge Detection", command=self.live_edge_detection,
    bg=self.btn_color, fg=self.fg_color, activebackground=self.highlight_color)
        self.live_edge_btn.pack(pady=5)


        # Edge detection button
        self.edge_btn = tk.Button(root, text="Custom Edge Detection", command=self.apply_edge_detection,
    bg=self.btn_color, fg=self.fg_color, activebackground=self.highlight_color)
        self.edge_btn.pack(pady=5)

        # Contour detection button
        self.contour_btn = tk.Button(root, text="Custom Contour Detection", command=self.apply_contour_detection,
    bg=self.btn_color, fg=self.fg_color, activebackground=self.highlight_color)
        self.contour_btn.pack(pady=5)

        # Save button
        self.save_btn = tk.Button(root, text="Save Result", command=self.save_result,
    bg=self.btn_color, fg=self.fg_color, activebackground=self.highlight_color)
        self.save_btn.pack(pady=5)

        # Image display area
        self.img_panel = tk.Label(root, bg=self.bg_color)
        self.img_panel.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            bgr_img = cv2.imread(file_path)
            if bgr_img is None:
                print("Invalid image")
                return
            self.image = bgr_img
            self.show_image(self.image)

    def capture_from_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam.")
            return

        print("📸 Press SPACE to capture image, ESC to cancel.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Webcam - Press SPACE to Capture", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC to exit
                break
            elif key == 32:  # SPACE to capture
                self.image = frame.copy()
                self.show_image(self.image)
                print("✅ Image captured from webcam.")
                break

        cap.release()
        cv2.destroyAllWindows()


    def apply_edge_detection(self):
        if self.image is not None:
            threshold = self.threshold_val.get()
            edge_map = custom_edge_detection(self.image, threshold=threshold)
            edge_bgr = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
            self.processed_img = edge_bgr
            self.show_image(edge_bgr)

    def live_edge_detection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam couldn't be opened.")
            return

        print("📡 Real-time edge detection started. Press ESC to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply custom edge detection
            edges = custom_edge_detection(frame, threshold=self.threshold_val.get())
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # Show original + edge side by side
            combined = np.hstack((frame, edges_bgr))
            cv2.imshow("Real-time Edge Detection", combined)

            key = cv2.waitKey(1)
            if key == 27:  # ESC to stop
                break

        cap.release()
        cv2.destroyAllWindows()


    def apply_contour_detection(self):
        if self.image is not None:
            threshold = self.threshold_val.get()
            edge_map = custom_edge_detection(self.image, threshold=threshold)
            contours = custom_contour_detection(edge_map)
            contour_img = draw_custom_contours(self.image, contours)
            self.processed_img = contour_img
            self.show_image(contour_img)

    def save_result(self):
        if self.processed_img is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
            if file_path:
                cv2.imwrite(file_path, self.processed_img)
                print(f"✅ Image saved to {file_path}")
        else:
            print("⚠️ No processed image to save!")


    def show_image(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil = image_pil.resize((600, 400))
        image_tk = ImageTk.PhotoImage(image_pil)
        self.img_panel.configure(image=image_tk)
        self.img_panel.image = image_tk

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeContourApp(root)
    root.mainloop()
