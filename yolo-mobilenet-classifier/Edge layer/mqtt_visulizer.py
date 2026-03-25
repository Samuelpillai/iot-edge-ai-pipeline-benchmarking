import json
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import paho.mqtt.client as mqtt
from collections import deque

# -- Config --
MAX_POINTS = 100
stop_animation = False

# -- Buffers --
fps_vals = deque(maxlen=MAX_POINTS)
cpu_vals = deque(maxlen=MAX_POINTS)
mem_vals = deque(maxlen=MAX_POINTS)
yolo_times = deque(maxlen=MAX_POINTS)
vector_times = deque(maxlen=MAX_POINTS)
classifier_times = deque(maxlen=MAX_POINTS)
total_times = deque(maxlen=MAX_POINTS)

# -- MQTT Callback --
def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        metrics = payload.get("metrics", {})

        fps_vals.append(metrics.get("fps", 0))
        cpu_vals.append(metrics.get("cpu_percent", 0))
        mem_vals.append(metrics.get("memory_mb", 0))
        yolo_times.append(metrics.get("yolo_time_ms", 0))
        vector_times.append(metrics.get("vector_time_ms", 0))
        classifier_times.append(metrics.get("classifier_time_ms", 0))
        total_times.append(metrics.get("total_time_ms", 0))
    except Exception as e:
        print(f"[ERROR] Failed to parse: {e}")

# -- MQTT Listener Thread --
def start_mqtt_listener():
    client = mqtt.Client()
    client.on_message = on_message
    # client.connect("172.19.85.154", 1883, 60)
    client.connect("10.67.150.154", 1883, 60)
    client.subscribe("inference/logsOfThirdP")
    client.loop_forever()

mqtt_thread = threading.Thread(target=start_mqtt_listener)
mqtt_thread.daemon = True
mqtt_thread.start()

# -- Matplotlib Setup --
fig, axs = plt.subplots(4, 2, figsize=(12, 8))
fig.suptitle("Pipeline Three — YOLO + Vectorizer + Classifier", fontsize=16)
fig.canvas.manager.set_window_title("Pipeline Three - Inference Dashboard")

def annotate(ax, values, unit=""):
    if values:
        val = values[-1]
        ax.text(
            0.95, 0.9, f"{val:.2f}{unit}",
            transform=ax.transAxes,
            fontsize=10, color="blue", ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6)
        )

def on_key_press(event):
    global stop_animation
    if event.key == 'g':
        print("🛑 'g' pressed — exiting plot.")
        stop_animation = True
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key_press)

def animate(i):
    if stop_animation:
        return

    axs[0, 0].clear()
    axs[0, 0].plot(fps_vals, label="FPS")
    axs[0, 0].set_title("Frames Per Second")
    if fps_vals:
        min_fps = min(fps_vals)
        max_fps = max(fps_vals)
        buffer = max(1, (max_fps - min_fps) * 0.5)
        axs[0, 0].set_ylim(min_fps - buffer, max_fps + buffer)
    annotate(axs[0, 0], fps_vals)

    axs[0, 1].clear()
    axs[0, 1].plot(cpu_vals, label="CPU %", color="orange")
    axs[0, 1].set_title("CPU Usage (%)")
    axs[0, 1].set_ylim(0, 500)
    annotate(axs[0, 1], cpu_vals, "%")

    axs[1, 0].clear()
    axs[1, 0].plot(mem_vals, label="Memory MB", color="green")
    axs[1, 0].set_title("Memory Usage (MB)")
    if mem_vals:
        min_mem = min(mem_vals)
        max_mem = max(mem_vals)
        buffer = max(5, (max_mem - min_mem) * 0.5)
        axs[1, 0].set_ylim(min_mem - buffer, max_mem + buffer)
    annotate(axs[1, 0], mem_vals, "MB")

    axs[1, 1].clear()
    axs[1, 1].plot(yolo_times, label="YOLO Time", color="gray")
    axs[1, 1].set_title("YOLO Inference Time (ms)")
    axs[1, 1].set_ylim(0, 500)
    annotate(axs[1, 1], yolo_times, "ms")

    axs[2, 0].clear()
    axs[2, 0].plot(vector_times, label="Vectorizer Time", color="purple")
    axs[2, 0].set_title("Vectorizer Time (ms)")
    axs[2, 0].set_ylim(0, 300)
    annotate(axs[2, 0], vector_times, "ms")

    axs[2, 1].clear()
    axs[2, 1].plot(classifier_times, label="Classifier Time", color="red")
    axs[2, 1].set_title("Classifier Time (ms)")
    axs[2, 1].set_ylim(0, 15)
    annotate(axs[2, 1], classifier_times, "ms")

    axs[3, 0].clear()
    axs[3, 0].plot(total_times, label="Total Time", color="black")
    axs[3, 0].set_title("Total Inference Time (ms)")
    axs[3, 0].set_ylim(0, 700)
    annotate(axs[3, 0], total_times, "ms")

    for ax in axs.flat:
        ax.legend()
        ax.grid(True)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.tight_layout()
plt.show()