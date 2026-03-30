import cv2
import pandas as pd
import argparse
from ultralytics import YOLO
import numpy as np
import ctypes

# ------------------------
# НАСТРОЙКИ
# ------------------------
EMPTY = 0
OCCUPIED = 1

STABILITY_FRAMES = 10       # сглаживание
MIN_STATE_DURATION = 1.0    # сек

# ------------------------
# ФУНКЦИЯ: пересечение bbox и ROI
# ------------------------
def intersects(box, roi):
    x1, y1, x2, y2 = box
    rx, ry, rw, rh = roi

    return not (x2 < rx or x1 > rx + rw or y2 < ry or y1 > ry + rh)

# ------------------------
# АРГУМЕНТЫ
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
args = parser.parse_args()

# ------------------------
# МОДЕЛЬ
# ------------------------
model = YOLO("yolov8n.pt")

# ------------------------
# ВИДЕО
# ------------------------
cap = cv2.VideoCapture(args.video)

if not cap.isOpened():
    raise Exception("Не удалось открыть видео")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ------------------------
# ПЕРВЫЙ КАДР
# ------------------------
ret, frame = cap.read()
if not ret:
    raise Exception("Ошибка чтения кадра")

# ------------------------
# ЭКРАН
# ------------------------
user32 = ctypes.windll.user32
screen_w = user32.GetSystemMetrics(0)
screen_h = user32.GetSystemMetrics(1)

h, w = frame.shape[:2]

scale = min(screen_w / w, screen_h / h)
scale = min(scale, 1.0)

resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

# ------------------------
# ROI
# ------------------------
cv2.namedWindow("Select table", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select table", resized.shape[1], resized.shape[0])

roi_scaled = cv2.selectROI("Select table", resized, False, False)
cv2.destroyAllWindows()

x, y, rw, rh = roi_scaled

# обратно в оригинал
x = int(x / scale)
y = int(y / scale)
rw = int(rw / scale)
rh = int(rh / scale)

roi = (x, y, rw, rh)

# ------------------------
# VIDEO WRITER
# ------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output/output.mp4", fourcc, fps, (width, height))

# ------------------------
# ЛОГИКА
# ------------------------
prev_state = EMPTY
state_buffer = []
events = []

frame_idx = 0
last_change_time = 0

# ------------------------
# ОСНОВНОЙ ЦИКЛ
# ------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    person_in_roi = False

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if intersects((x1, y1, x2, y2), roi):
            person_in_roi = True
            break

    # ------------------------
    # СГЛАЖИВАНИЕ
    # ------------------------
    raw_state = OCCUPIED if person_in_roi else EMPTY

    state_buffer.append(raw_state)
    if len(state_buffer) > STABILITY_FRAMES:
        state_buffer.pop(0)

    if sum(state_buffer) > len(state_buffer) / 2:
        current_state = OCCUPIED
    else:
        current_state = EMPTY

    timestamp = frame_idx / fps

    # ------------------------
    # СОБЫТИЯ (с фильтром)
    # ------------------------
    if prev_state != current_state:
        if timestamp - last_change_time > MIN_STATE_DURATION:

            if prev_state == EMPTY and current_state == OCCUPIED:
                events.append({"time": timestamp, "event": "APPROACH"})
                print(f"[{timestamp:.2f}] APPROACH")

            if prev_state == OCCUPIED and current_state == EMPTY:
                events.append({"time": timestamp, "event": "EMPTY"})
                print(f"[{timestamp:.2f}] EMPTY")

            prev_state = current_state
            last_change_time = timestamp

    # ------------------------
    # ВИЗУАЛИЗАЦИЯ
    # ------------------------
    color = (0, 255, 0) if current_state == EMPTY else (0, 0, 255)

    cv2.rectangle(frame, (x, y), (x + rw, y + rh), color, 3)

    text = "EMPTY" if current_state == EMPTY else "OCCUPIED"
    cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    out.write(frame)

    frame_idx += 1

# ------------------------
# АНАЛИТИКА
# ------------------------
df = pd.DataFrame(events)

delays = []
last_empty = None

for _, row in df.iterrows():
    if row["event"] == "EMPTY":
        last_empty = row["time"]

    if row["event"] == "APPROACH" and last_empty is not None:
        delays.append(row["time"] - last_empty)
        last_empty = None

if delays:
    print("\n=== RESULT ===")
    print(f"Средняя задержка: {np.mean(delays):.2f} сек")
else:
    print("Недостаточно данных")

# ------------------------
# ЗАВЕРШЕНИЕ
# ------------------------
cap.release()
out.release()
cv2.destroyAllWindows()