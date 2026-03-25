# classifier_subscriber.py — receives vector and classifies
import time
import json
import numpy as np
import paho.mqtt.client as mqtt
import joblib
import psutil
import os
import mlflow
import atexit
from datetime import datetime

# Load classifier
clf = joblib.load("./faceM.joblib")
process = psutil.Process(os.getpid())
frame_count = 0
start_time = time.time()

# Log publisher
log_client = mqtt.Client()
log_client.connect("10.67.150.154", 1883, 60)
log_client.loop_start()

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
mlflow.set_experiment("pipeline on IoT")
mlflow.start_run(run_name=f"pipeline-2 | yolo/vector | {datetime.now().strftime('%Y%m%d_%H%M%S')}")

def on_message(client, userdata, msg):
    global frame_count, start_time

    total_start = time.perf_counter()

    # Receive vector + IoT metrics
    payload = json.loads(msg.payload.decode())
    vector = np.array(payload["vector"])
    label = payload.get("label", "unknown")
    bbox = payload.get("bbox")

    # Received times from IoT
    yolo_time = payload.get("yolo_time_ms", None)
    vector_time = payload.get("vector_time_ms", None)

    # Classification
    t1 = time.perf_counter()
    probs = clf.predict_proba([vector])[0]
    t2 = time.perf_counter()

    predicted = "unknown" if max(probs) < 0.5 else clf.classes_[np.argmax(probs)]
    total_end = time.perf_counter()

    # System metrics
    cpu = process.cpu_percent(interval=0)
    mem = process.memory_info().rss / 1024 / 1024  # MB
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    classifier_time = (t2 - t1)*1000
    total_time = (total_end - total_start)*1000

    # Print formatted log
    print(f"Prediction: {predicted} — Prob: {max(probs):.2f}")
    print(
        f"[EDGE] Label: {predicted} |"
        f" Classifier: {(t2 - t1)*1000:.2f}ms |"
        f" Total: {(total_end - total_start)*1000:.2f}ms |"
        f" FPS: {fps:.2f} | CPU: {cpu:.2f}% | RAM: {mem:.2f}MB"
    )

    # Append vector + label to shared .jsonl file
    try:
        with open("/Users/sam/CSC8199/OpenCV/RPiPipeline/vector_inference.jsonl", "a") as f:
            f.write(json.dumps({
                "vector": vector.tolist(),
                "label": predicted
            }) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to write to vector_inference.jsonl: {e}")

    #Final benchmark payload
    log_payload = {
        "label":predicted,
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

    log_client.publish("inference/logsOfSecP", json.dumps(log_payload))

    try:
        mlflow.log_param("classifier_model", "FaceM.joblib")
        mlflow.log_param("vectorizer_script", "mobilenet_vectorizer.py")
        mlflow.log_param("yolo_model", "yolov8n_169-4.pt")
        mlflow.log_param("yolo_script", "yolo_pub.py")

        mlflow.log_metric("yolo_time_ms", yolo_time)
        mlflow.log_metric("vector_time_ms", vector_time)
        mlflow.log_metric("classifier_time_ms", classifier_time)
        mlflow.log_metric("total_time_ms", total_time)
        mlflow.log_metric("fps", fps)
        mlflow.log_metric("cpu_percent", cpu)
        mlflow.log_metric("memory_mb", mem)

        # Optional: if you compute accuracy at end, log once (not per frame)
        # mlflow.log_metric("accuracy", accuracy)
    except Exception as e:
        print(f"[MLFLOW ERROR] {e}")

# MQTT Setup
client = mqtt.Client()
client.on_message = on_message
client.connect("10.67.150.154", 1883, 60)
client.subscribe("yolo/vector")
client.loop_forever()