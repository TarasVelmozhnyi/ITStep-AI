import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Шляхи ---
desktop_path = os.path.expanduser("~/Desktop")
slides_folder = os.path.join(desktop_path, "slides")

# Перевірка наявності папки
if not os.path.exists(slides_folder):
    print(f"Помилка: Папка '{slides_folder}' не знайдена!")
    exit(1)

# ================= НАЛАШТУВАННЯ =================
SLIDE_W, SLIDE_H = 640, 480  # Фіксований розмір для всіх слайдів
GESTURE_DELAY = 1.0

COLORS = [
    (0, 0, 255),  # червоний
    (0, 255, 0),  # зелений
    (255, 0, 0)  # синій
]
color_id = 0

# Завантаження слайдів з ОБОВ'ЯЗКОВИМ зміненням розміру
slide_files = [f for f in os.listdir(slides_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
if not slide_files:
    print(f"Помилка: У папці '{slides_folder}' не знайдено зображень!")
    exit(1)

# Завантаження та зміна розміру зображень слайдів
slides = []
for slide_file in slide_files:
    slide_path = os.path.join(slides_folder, slide_file)
    slide_img = cv2.imread(slide_path)
    if slide_img is not None:
        # ЗМІНА РОЗМІРУ ДО ФІКСОВАНОГО
        slide_img = cv2.resize(slide_img, (SLIDE_W, SLIDE_H))
        slides.append(slide_img)
    else:
        print(f"Попередження: Не вдалося завантажити {slide_file}")

if not slides:
    print("Помилка: Не вдалося завантажити жодного слайду!")
    exit(1)

# ================= MediaPipe =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= СТАН =================
slide_id = 0
canvas = np.zeros((SLIDE_H, SLIDE_W, 3), dtype=np.uint8)

drawing = False
eraser = False
last_point = None
last_gesture_time = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Камера не відкрилася")
    exit()


# ================= ФУНКЦІЯ ПАЛЬЦІ =================
def fingers_up(lm):
    fingers = [0, 0, 0, 0, 0]

    # великий (для правої руки)
    fingers[0] = lm[4].x < lm[3].x

    # інші пальці
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for i in range(4):
        fingers[i + 1] = lm[tips[i]].y < lm[pips[i]].y

    return fingers


# ================= ГОЛОВНИЙ ЦИКЛ =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        f = fingers_up(lm)
        now = time.time()

        # ---------- ЖЕСТИ ----------
        if now - last_gesture_time > GESTURE_DELAY:

            if sum(f) == 5:  # ✋
                canvas[:] = 0
                slide_id = 0
                drawing = False
                eraser = False
                last_gesture_time = now

            elif f == [0, 1, 0, 0, 0]:  # ☝️
                slide_id = (slide_id + 1) % len(slides)
                last_gesture_time = now

            elif f == [1, 0, 0, 0, 0]:  # 👍
                slide_id = (slide_id - 1) % len(slides)
                last_gesture_time = now

            elif f == [0, 1, 1, 0, 0]:  # ✌️
                drawing = True
                eraser = False

            elif f == [1, 1, 1, 0, 0]:  # ✌️ + 👍
                color_id = (color_id + 1) % len(COLORS)
                last_gesture_time = now

            elif sum(f) == 0:  # ✊ гумка
                drawing = True
                eraser = True

        # ---------- МАЛЮВАННЯ ----------
        if drawing and f[1] == 1:
            x = int(lm[8].x * SLIDE_W)
            y = int(lm[8].y * SLIDE_H)

            if last_point:
                color = (0, 0, 0) if eraser else COLORS[color_id]
                thickness = 30 if eraser else 5
                cv2.line(canvas, last_point, (x, y), color, thickness)

            last_point = (x, y)
        else:
            last_point = None

    # ================= НАКЛАДАННЯ CANVAS =================
    slide = slides[slide_id].copy()
    mask = np.any(canvas != [0, 0, 0], axis=2)

    # ТЕПЕР ВСЕ ПРАЦЮЄ, БО ВСІ СЛАЙДИ МАЮТЬ ОДНАКОВИЙ РОЗМІР
    slide[mask] = canvas[mask]

    # ================= ІНФОРМАЦІЙНЕ ПАНЕЛЬ =================
    # Додай інформацію про поточний стан
    info_text = f"Slide: {slide_id + 1}/{len(slides)} | "
    info_text += f"Color: {COLORS[color_id]} | "
    info_text += f"Draw: {'ON' if drawing else 'OFF'} | "
    info_text += f"Eraser: {'ON' if eraser else 'OFF'}"

    cv2.putText(slide, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(slide, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    cv2.imshow("Presentation", slide)
    cv2.imshow("Canvas", canvas)  # Можеш переглянути окремо canvas
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= ОЧИЩЕННЯ =================
cap.release()
cv2.destroyAllWindows()