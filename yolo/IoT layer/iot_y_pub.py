import cv2
import json
import time
import psutil
import os
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from picamera2 import Picamera2

label_results = {}

def on_connect(client, userdata, flags, rc):
    print("[MQTT] Connected:", rc)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        bbox = tuple(data["bbox"])
        label = data["label"]
        label_results[bbox] = label
    except Exception as e:
        print(f"[ERROR] Failed to handle message: {e}")

# Setup MQTT
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("10.113.51.154", 1883, 60)
client.loop_start()

# Load model
model = YOLO("./yolov8n.pt")
print("[INFO] YOLO model loaded.")

# --- ✅ Camera Setup Using Picamera2 ---
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Monitoring
process = psutil.Process(os.getpid())
frame_count = 0
start_time = time.time()

while True:
    loop_start = time.time()
    print("[DEBUG] Reading from camera...")

    try:
        im = picam2.capture_array()
        frame = im.copy()
    except Exception as e:
        print(f"[ERROR] Camera read failed: {e}")
        break

    # YOLO inference
    t0 = time.perf_counter()
    results = model(frame)
    t1 = time.perf_counter()
    yolo_time = (t1 - t0) * 1000
    print(f"[DEBUG] YOLO inference done. Found {len(results)} result(s).")

    for r in results:
        if not r.boxes:
            print("[DEBUG] No boxes detected.")
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bbox = (x1, y1, x2, y2)
            label = model.names[int(box.cls[0])]

            # YOLO detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            payload = {
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "shape": frame.shape,
                "yolo_time_ms": yolo_time
            }
            client.publish("yolo/bbox", json.dumps(payload))

    # CPU & Memory usage
    cpu = process.cpu_percent(interval=0)
    mem = process.memory_info().rss / 1024 / 1024  # in MB

    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    print(f"[YOLO PUB] Inference: {yolo_time:.2f}ms | FPS: {fps:.2f} | CPU: {cpu:.2f}% | RAM: {mem:.2f}MB")

    cv2.imshow("YOLO Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean shutdown
client.loop_stop()
client.disconnect()
picam2.stop()
cv2.destroyAllWindows()