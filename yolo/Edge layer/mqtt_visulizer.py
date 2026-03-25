import json
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import paho.mqtt.client as mqtt
from collections import deque

# Keep last 100 points for smoother animation
MAX_POINTS = 100

# Buffers
fps_vals = deque(maxlen=MAX_POINTS)
cpu_vals = deque(maxlen=MAX_POINTS)
mem_vals = deque(maxlen=MAX_POINTS)
vector_times = deque(maxlen=MAX_POINTS)
class_times = deque(maxlen=MAX_POINTS)
total_times = deque(maxlen=MAX_POINTS)
labels = deque(maxlen=MAX_POINTS)

# MQTT Callback
def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        metrics = payload.get("metrics", {})
        label = payload.get("label", "N/A")

        labels.append(label)
        fps_vals.append(metrics.get("fps", 0))
        cpu_vals.append(metrics.get("cpu_percent", 0))
        mem_vals.append(metrics.get("memory_mb", 0))
        vector_times.append(metrics.get("vector_time_ms", 0))
        class_times.append(metrics.get("classifier_time_ms", 0))
        total_times.append(metrics.get("total_time_ms", 0))
    except Exception as e:
        print(f"[ERROR] Failed to parse message: {e}")

# MQTT Setup
def start_mqtt_listener():
    client = mqtt.Client()
    client.on_message = on_message
    # client.connect("172.19.85.154", 1883, 60)
    client.connect("10.67.150.154", 1883, 60)
    client.subscribe("inference/logsOfOneP")
    client.loop_forever()

# Start MQTT in background
mqtt_thread = threading.Thread(target=start_mqtt_listener)
mqtt_thread.daemon = True
mqtt_thread.start()

# Setup Matplotlib
fig, axs = plt.subplots(3, 2, figsize=(12, 8))
fig.suptitle("Live Inference Metrics", fontsize=16)
fig.canvas.manager.set_window_title("Pipeline One - Inference Dashboard")

def annotate(ax, values, unit=""):
    if values:
        val = values[-1]
        ax.text(
            0.95, 0.9, f"{val:.2f}{unit}",
            transform=ax.transAxes,
            fontsize=10, color="blue", ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6)
        )

# Global flag to stop animation
stop_animation = False

def on_key_press(event):
    global stop_animation
    if event.key == 'g':
        print("🛑 'g' pressed — exiting plot.")
        stop_animation = True
        plt.close(fig)  # Close the plot window

# Register key press event
fig.canvas.mpl_connect('key_press_event', on_key_press)

def animate(i):
    axs[0, 0].clear()
    axs[0, 0].plot(fps_vals, label="FPS")
    axs[0, 0].set_title("Frames Per Second")
    axs[0, 0].set_ylim(0, 10)
    annotate(axs[0, 0], fps_vals)

    axs[0, 1].clear()
    axs[0, 1].plot(cpu_vals, label="CPU %", color="orange")
    axs[0, 1].set_title("CPU Usage (%)")
    axs[0, 1].set_ylim(0, 100)
    annotate(axs[0, 1], cpu_vals, "%")

    axs[1, 0].clear()
    axs[1, 0].plot(mem_vals, label="Memory MB", color="green")
    axs[1, 0].set_title("Memory Usage (MB)")
    axs[1, 0].set_ylim(200, 1000)
    annotate(axs[1, 0], mem_vals, "MB")

    axs[1, 1].clear()
    axs[1, 1].plot(vector_times, label="Vector Time", color="purple")
    axs[1, 1].set_title("Vectorizer Time (ms)")
    axs[1, 1].set_ylim(0, 100)
    annotate(axs[1, 1], vector_times, "ms")

    axs[2, 0].clear()
    axs[2, 0].plot(class_times, label="Classifier Time", color="red")
    axs[2, 0].set_title("Classifier Time (ms)")
    axs[2, 0].set_ylim(0, 20)
    annotate(axs[2, 0], class_times, "ms")

    axs[2, 1].clear()
    axs[2, 1].plot(total_times, label="Total Time", color="black")
    axs[2, 1].set_title("Total Inference Time (ms)")
    axs[2, 1].set_ylim(0, 150)
    annotate(axs[2, 1], total_times, "ms")

    for ax in axs.flat:
        ax.legend()
        ax.grid(True)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.tight_layout()
plt.show()