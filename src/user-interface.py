import cv2
import threading
import time
import numpy as np
from IPython.display import display, clear_output
import ipywidgets as widgets
from vision_module import VisionModule

# Initialize Vision Module (same-folder best.pt)
vision = VisionModule()

# UI widgets
start_btn = widgets.Button(description="▶ Start Camera", button_style='success')
stop_btn  = widgets.Button(description="⛔ Stop", button_style='danger')
save_btn  = widgets.Button(description="💾 Save Frame")
mode_dropdown = widgets.Dropdown(options=["Object Detection", "Color Only"], description="Mode:")
output_area = widgets.Output()

display(widgets.VBox([widgets.HBox([start_btn, stop_btn, save_btn]),
                      mode_dropdown,
                      output_area]))

# Camera control
running = False
cap = None
current_frame = None

def camera_loop():
    global running, cap, current_frame
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        current_frame = frame.copy()

        if mode_dropdown.value == "Object Detection":
            detections = vision.detect_objects(frame)
            color_bins = vision.classify_by_color(frame, detections)
            annotated = vision.draw_detections(frame, detections, show_colors=True)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0,50,50), (180,255,255))
            annotated = cv2.bitwise_and(frame, frame, mask=mask)

        _, jpeg = cv2.imencode('.jpg', annotated)
        img_bytes = jpeg.tobytes()
        with output_area:
            clear_output(wait=True)
            display(widgets.Image(value=img_bytes, format='jpeg'))

        time.sleep(0.03)

def start_camera(b):
    global running, cap
    if running: return
    cap = cv2.VideoCapture(0)
    running = True
    threading.Thread(target=camera_loop).start()
    print("Camera started")

def stop_camera(b):
    global running, cap
    running = False
    if cap:
        cap.release()
    print("Camera stopped")

def save_frame(b):
    if current_frame is not None:
        filename = f"saved_{int(time.time())}.jpg"
        cv2.imwrite(filename, current_frame)
        print(f"Saved: {filename}")

start_btn.on_click(start_camera)
stop_btn.on_click(stop_camera)
save_btn.on_click(save_frame)
