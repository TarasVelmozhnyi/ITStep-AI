# terminal
# cd /Users/tarasvelmozhnyi/Documents/
# python3 hand_presentation.py

import cv2
import mediapipe as mp
import numpy as np
import os
import time


#   розмір
W, H = 680, 420

cv2.namedWindow("Presentation")
cv2.namedWindow("Canvas")
cv2.namedWindow("Camera")

# MediaPipe

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

#  кольори  # BGR
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
color_index = 0

# # слайди

slides_path = "/Users/tarasvelmozhnyi/Desktop/slides"

# файли отримати
slide_files = []
for file in os.listdir(slides_path):
    name_lower = file.lower()
    if name_lower.endswith(('.png', '.jpg', '.jpeg')):
        slide_files.append(file)

#  сортування (працює, якщо назви одн

slide_files.sort()

slides = []
for filename in slide_files:
    img = cv2.imread(os.path.join(slides_path, filename))
    img = cv2.resize(img, (W, H))
    slides.append(img)

# - Стан застосунку ---

current_slide = 0
canvas = np.zeros((H, W, 3), dtype=np.uint8)
last_point = None
gesture_timer = 0
nav_time = 0

cap = cv2.VideoCapture(0)

print("Запускаємо : Presentation, Canvas, Camera.")


def compose_slide(slide, drawing):
    """Об'єднує слайд з малюнком"""

    result = slide.copy()
    mask = np.any(drawing > 10, axis=2)
    result[mask] = drawing[mask]

    return result


# Розташування вікон
cv2.moveWindow("Presentation", 50, 50)  # ліворуч зверху
cv2.moveWindow("Canvas", W + 100, 50)  # праворуч зверху
cv2.moveWindow("Camera", 50, H + 100)  # ліворуч знизу

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    frame_draw = cv2.resize(frame, (W, H))
    rgb_frame = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    fingers = [0, 0, 0, 0, 0]
    curr_time = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_draw, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark

            # Визначаємо які пальці
            if lm[4].x > lm[3].x + 0.01: fingers[0] = 1  # Великий
            for i, tip in enumerate([8, 12, 16, 20]):
                if lm[tip].y < lm[tip - 2].y: fingers[i + 1] = 1  # Інші

            ix, iy = int(lm[8].x * W), int(lm[8].y * H)

            #  Долоня - Очистити і скинути на початок
            if sum(fingers) == 5:
                canvas = np.zeros_like(canvas)
                current_slide = 0

            #  Вказівний + Середній - Малювання
            elif fingers[1] and fingers[2] and not fingers[0]:
                if last_point is not None:
                    cv2.line(canvas, last_point, (ix, iy), colors[color_index], 8)
                last_point = (ix, iy)
                gesture_timer = 0

            #  Вказівний + Середній + Великий - Зміна кольору (затримка 1с)
            elif fingers[0] and fingers[1] and fingers[2]:
                if gesture_timer == 0:
                    gesture_timer = curr_time
                elif curr_time - gesture_timer > 1.0:
                    color_index = (color_index + 1) % len(colors)
                    gesture_timer = curr_time
                    print("Колір змінено!")
                last_point = None

            #  Вказівний (один) - Наступний слайд
            elif fingers[1] and sum(fingers) == 1:
                if curr_time - nav_time > 0.8:
                    current_slide = (current_slide + 1) % len(slides)
                    nav_time = curr_time
                last_point = None

            #  Великий (один) - Попередній слайд
            elif fingers[0] and sum(fingers) == 1:
                if curr_time - nav_time > 0.8:
                    current_slide = (current_slide - 1) % len(slides)
                    nav_time = curr_time
                last_point = None

            #  Вказівний + Мізинець - Гумка
            elif fingers[1] and fingers[4]:
                cv2.circle(canvas, (ix, iy), 30, (0, 0, 0), -1)
                last_point = None

            else:
                last_point = None
                gesture_timer = 0

    slide_img = compose_slide(slides[current_slide], canvas)

    cv2.imshow("Presentation", slide_img)  # 1. Результат
    cv2.imshow("Canvas", canvas)  # 2. Полотно (чорне)
    cv2.imshow("Camera", frame_draw)  # 3. Камера


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()