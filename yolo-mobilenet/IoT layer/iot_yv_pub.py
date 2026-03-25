# iot_vectorizer.py — minimal vector-only MQTT publisher
import cv2
import json
import time
import psutil
import os
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from mobilenet_vectorizer import MobileNetVectorizer
from picamera2 import Picamera2

def on_connect(client, userdata, flags, rc):
    print("[MQTT] Connected:", rc)

# MQTT clients
vector_pub = mqtt.Client()
vector_pub.on_connect = on_connect
vector_pub.connect("10.113.51.154", 1883, 60)
vector_pub.loop_start()

log_pub = mqtt.Client()
log_pub.on_connect = on_connect
log_pub.connect("10.113.51.154", 1883, 60)
log_pub.loop_start()

# Load models
yolo = YOLO("./yolov8n_169-4.pt")
print("[INFO] YOLO model loaded.")
vectorizer = MobileNetVectorizer()

# cap = cv2.VideoCapture(0)

# Pi camera Setup
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

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
    results = yolo(frame)
    t1 = time.perf_counter()
    yolo_time = (t1 - t0) * 1000
    print(f"[DEBUG] YOLO inference done. Found {len(results)} result(s).")

    # Benchmark info (always computed)
    cpu = process.cpu_percent(interval=0)
    mem = process.memory_info().rss / 1024 / 1024
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    # If detections exist, publish vectors
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = yolo.names[int(box.cls[0])]

            # YOLO detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            t2 = time.perf_counter()
            vector = vectorizer.get_vector(crop)
            t3 = time.perf_counter()

            vector_pub.publish("yolo/vector", json.dumps({
                "vector": vector,
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "yolo_time_ms": yolo_time,
                "vector_time_ms": (t3 - t2)*1000
            }))

            # Also publish vector timing log
            log_payload = {
                "stage": "iot_vectorizer",
                "metrics": {
                    "yolo_time_ms": yolo_time,
                    "vector_time_ms": (t3 - t2) * 1000,
                    "cpu_percent": cpu,
                    "memory_mb": mem,
                    "fps": fps
                }
            }
            log_pub.publish("inference/logsOfSecP", json.dumps(log_payload))

    # Now safe to print fps, even if no detection
    print(f"[IOT] Inference: {yolo_time:.2f}ms | FPS: {fps:.2f} | CPU: {cpu:.2f}% | RAM: {mem:.2f}MB")

    cv2.imshow("YOLO Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean shutdown
vector_pub.loop_stop()
log_pub.loop_stop()
vector_pub.disconnect()
log_pub.disconnect()
picam2.stop()
cv2.destroyAllWindows()