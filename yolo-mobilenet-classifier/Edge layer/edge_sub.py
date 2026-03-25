import json
import mlflow
import atexit
import paho.mqtt.client as mqtt
from datetime import datetime

# === MLflow Setup ===
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("pipeline on IoT")
mlflow.start_run(run_name=f"pipeline-3 | yolo/vector/class | {datetime.now().strftime('%Y%m%d_%H%M%S')}")

@atexit.register
def end_run():
    mlflow.end_run()

output_file = "/Users/sam/CSC8199/OpenCV/RPiPipeline/vector_inference.jsonl"

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode()

    try:
        data = json.loads(payload)

        # === Handle vector data ===
        if topic == "yolo/vector/class":
            print("📦 Vector received:")
            print(json.dumps(data, indent=2))
            with open(output_file, "a") as f:
                f.write(json.dumps({
                    "vector": list(data["vector"]),
                    "label": str(data["label"])
                }) + "\n")
            print(" Saved to vector_inference.jsonl")

        # === Handle metrics data ===
        elif topic == "inference/logsOfThirdP":
            metrics = data.get("metrics", {})

            # Log params once (or mock here)
            mlflow.log_param("classifier_model", "FaceM.joblib")
            mlflow.log_param("vectorizer_script", "mobilenet_vectorizer.py")
            mlflow.log_param("yolo_model", "yolov8n_169-4.pt")
            mlflow.log_param("yolo_script", "yolo_pub.py")

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            print(f"[MLFLOW] Logged: {metrics}")

    except Exception as e:
        print(f" Error handling message on topic {topic}: {e}")

# === Single MQTT client handles both topics ===
client = mqtt.Client(protocol=mqtt.MQTTv311)
client.on_message = on_message
client.connect("10.67.150.154", 1883, 60)
client.subscribe("yolo/vector/class")
client.subscribe("inference/logsOfThirdP")
client.loop_forever()