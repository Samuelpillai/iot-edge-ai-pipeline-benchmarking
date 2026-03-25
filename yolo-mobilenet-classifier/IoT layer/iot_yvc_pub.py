import cv2
import json
import time
import psutil
import os
import paho.mqtt.client as mqtt
import numpy as np
import joblib
from ultralytics import YOLO
from mobilenet_vectorizer import MobileNetVectorizer
from picamera2 import Picamera2

def on_connect(client, userdata, flags, rc):
    print("[MQTT] Connected:", rc)

# MQTT clients
label_pub = mqtt.Client()
label_pub.connect("10.113.51.154", 1883, 60)
label_pub.on_connect = on_connect
label_pub.loop_start()

log_pub = mqtt.Client()
log_pub.connect("10.113.51.154", 1883, 60)
log_pub.on_connect = on_connect
log_pub.loop_start()

# Load models
yolo = YOLO("./yolov8n_169-4.pt")
print("[INFO] YOLO model loaded.")
vectorizer = MobileNetVectorizer()
clf = joblib.load("./model.joblib")

# cap = cv2.VideoCapture(0)

# Pi camera setup
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
    results = yolo(frame)
    t1 = time.perf_counter()
    yolo_time = (t1 - t0) * 1000

    # Benchmark info
    cpu = process.cpu_percent(interval=0)
    mem = process.memory_info().rss / 1024 / 1024
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    # Process detections
    for r in results:
        if not r.boxes:
            print("[DEBUG] No boxes detected.")
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            base_label = yolo.names[int(box.cls[0])]

            # YOLO detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, base_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Vectorization
            t2 = time.perf_counter()
            vector = vectorizer.get_vector(crop)
            t3 = time.perf_counter()
            vector_time = (t3 - t2) * 1000

            # Classification
            t4 = time.perf_counter()
            probs = clf.predict_proba([vector])[0]
            predicted_label = "unknown" if max(probs) < 0.5 else clf.classes_[np.argmax(probs)]
            t5 = time.perf_counter()
            classifier_time = (t5 - t4) * 1000
            total_time = (t5 - t0) * 1000

            # Publish to MQTT
            print("[DEBUG] Publishing vector to topic yolo/vector/class")
            label_pub.publish("yolo/vector/class", json.dumps({
                "vector": vector,
                "label": predicted_label
            }))

            # Publish benchmark logs
            log_payload = {
                "label": predicted_label,
                "metrics": {
                    "yolo_time_ms": yolo_time,
                    "vector_time_ms": vector_time,
                    "classifier_time_ms": classifier_time,
                    "total_time_ms": total_time,
                    "cpu_percent": cpu,
                    "memory_mb": mem,
                    "fps": fps
                }
            }
            log_pub.publish("inference/logsOfThirdP", json.dumps(log_payload))

            print(json.dumps({"vector": vector[:5], "label": predicted_label}, indent=2))
            print(f"[IOT] Label: {predicted_label} | Conf: {max(probs):.2f} | "
                  f"YOLO: {yolo_time:.2f}ms | Vector: {vector_time:.2f}ms | "
                  f"Classifier: {classifier_time:.2f}ms | Total: {total_time:.2f}ms")

    # Visual
    cv2.imshow("YOLO Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean shutdown
label_pub.loop_stop()
log_pub.loop_stop()
label_pub.disconnect()
log_pub.disconnect()
picam2.stop()
cv2.destroyAllWindows()