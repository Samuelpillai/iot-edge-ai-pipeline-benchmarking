# vectorizer_classifier.py
import time
import cv2
import json
import numpy as np
import paho.mqtt.client as mqtt
import psutil
import os
import joblib
import mlflow
import atexit
from datetime import datetime
from mobilenet_vectorizer import MobileNetVectorizer

# Initialize vectorizer and classifier
vectorizer = MobileNetVectorizer()
clf = joblib.load("./faceM.joblib")
video = cv2.VideoCapture(0)

# Setup main MQTT client (subscriber)
client = mqtt.Client()

# Setup separate MQTT client for publishing logs
log_client = mqtt.Client()
log_client.connect("10.67.150.154", 1883, 60)
log_client.loop_start()

process = psutil.Process(os.getpid())
frame_count = 0
start_time = time.time()

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
mlflow.set_experiment("pipeline on IoT")
mlflow.start_run(run_name=f"pipeline-1 | yolo | {datetime.now().strftime('%Y%m%d_%H%M%S')}")

def on_message(client, userdata, msg):
    global frame_count, start_time
    total_start = time.perf_counter()

    payload = json.loads(msg.payload.decode())
    x1, y1, x2, y2 = payload["bbox"]

    ret, frame = video.read()
    if not ret:
        return

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return

    # Extract 1280-dim feature vector
    t1 = time.perf_counter()
    vector = vectorizer.get_vector(crop)
    t2 = time.perf_counter()

    # Predict label
    probs = clf.predict_proba([vector])[0]
    t3 = time.perf_counter()

    if max(probs) < 0.5:
        label = "unknown"
    else:
        label = clf.classes_[np.argmax(probs)]

    total_end = time.perf_counter()

    # CPU & RAM
    cpu = process.cpu_percent(interval=0)
    mem = process.memory_info().rss / 1024 / 1024  # MB
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    # Print for human monitoring
    print(f"Prediction: {label} — Prob: {max(probs):.2f}")
    print(
        f"[EDGE] Label: {label} |"
        f" Vector: {(t2 - t1)*1000:.2f}ms |"
        f" Classifier: {(t3 - t2)*1000:.2f}ms |"
        f" Total: {(total_end - total_start)*1000:.2f}ms |"
        f" FPS: {fps:.2f} | CPU: {cpu:.2f}% | RAM: {mem:.2f}MB"
    )

    payload = json.loads(msg.payload.decode())
    yolo_time_ms = payload.get("yolo_time_ms", None)

    # Append vector + label to shared .jsonl file
    try:
        with open("/Users/sam/CSC8199/OpenCV/RPiPipeline/vector_inference.jsonl", "a") as f:
            f.write(json.dumps({
                "vector": vector,
                "label": label
            }) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to write to vector_inference.jsonl: {e}")

    # Send only benchmark log (NO vector, NO probs)
    log_payload = {
        "label": label,
        "metrics": {
            "yolo_time_ms": yolo_time_ms,
            "vector_time_ms": (t2 - t1) * 1000,
            "classifier_time_ms": (t3 - t2) * 1000,
            "total_time_ms": (total_end - total_start) * 1000,
            "cpu_percent": cpu,
            "memory_mb": mem,
            "fps": fps
        }
    }
    log_client.publish("inference/logsOfOneP", json.dumps(log_payload))

    label_payload = {
        "label":label,
        "bbox": [x1, y1, x2, y2]
    }
    log_client.publish("inference/label", json.dumps(label_payload))

    try:
        mlflow.log_param("classifier_model", "FaceM.joblib")
        mlflow.log_param("vectorizer_script", "mobilenet_vectorizer.py")
        mlflow.log_param("yolo_model", "yolov8n_169-4.pt")
        mlflow.log_param("yolo_script", "yolo_pub.py")

        if yolo_time_ms is not None:
            mlflow.log_metric("yolo_time_ms", yolo_time_ms)
        mlflow.log_metric("vector_time_ms", (t2 - t1) * 1000)
        mlflow.log_metric("classifier_time_ms", (t3 - t2) * 1000)
        mlflow.log_metric("total_time_ms", (total_end - total_start) * 1000)
        mlflow.log_metric("fps", fps)
        mlflow.log_metric("cpu_percent", cpu)
        mlflow.log_metric("memory_mb", mem)

        # Optional: if you compute accuracy at end, log once (not per frame)
        # mlflow.log_metric("accuracy", accuracy)
    except Exception as e:
        print(f"[MLFLOW ERROR] {e}")

# Subscribe to YOLO bounding box topic
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.subscribe("yolo/bbox")
atexit.register(lambda : mlflow.end_run())
client.loop_forever()