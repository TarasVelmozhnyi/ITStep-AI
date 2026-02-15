# # ================== DISABLE WARNINGS ==================
# import os
# import warnings
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# warnings.filterwarnings("ignore")
#
# # ================== IMPORTS ==================
# import cv2
# import mediapipe as mp
# import numpy as np
# import time
#
# # ================== MEDIAPIPE ==================
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     max_num_hands=1,
#     min_detection_confidence=0.9,
#     min_tracking_confidence=0.8
# )
# mp_draw = mp.solutions.drawing_utils
#
# # ================== SLIDES ==================
# slides_folder = os.path.expanduser("~/Desktop/slides")
# slide_files = sorted([
#     f for f in os.listdir(slides_folder)
#     if f.lower().endswith(('.png', '.jpg', '.jpeg'))
# ])
#
# slides = [cv2.imread(os.path.join(slides_folder, f)) for f in slide_files]
#
# SLIDE_W, SLIDE_H = 600, 400
# slides = [cv2.resize(s, (SLIDE_W, SLIDE_H)) for s in slides]
#
# # ================== STATE ==================
# current_slide = 0
# canvas = np.zeros((SLIDE_H, SLIDE_W, 3), dtype=np.uint8)
#
# drawing = False
# last_point = None
#
# colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # red, green, blue
# color_index = 0
# color = colors[color_index]
#
# gesture_start_time = None
#
# # ================== CAMERA ==================
# cap = cv2.VideoCapture(0)
#
# # ================== MAIN LOOP ==================
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break
#
#     frame = cv2.flip(frame, 1)
#     frame = cv2.medianBlur(frame, 5)
#
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb)
#
#     fingers = [0, 0, 0, 0, 0]  # thumb, index, middle, ring, pinky
#     now = time.time()
#
#     if results.multi_hand_landmarks:
#         hand = results.multi_hand_landmarks[0]
#         lm = hand.landmark
#         mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
#
#         # ---- finger detection ----
#         if lm[4].x < lm[3].x:
#             fingers[0] = 1
#
#         tips = [8, 12, 16, 20]
#         pips = [6, 10, 14, 18]
#
#         for i in range(4):
#             if lm[tips[i]].y < lm[pips[i]].y:
#                 fingers[i + 1] = 1
#
#         # ---- gestures ----
#
#         # ✋ Open palm — start presentation + clear canvas
#         if sum(fingers) == 5:
#             canvas[:] = 0
#             drawing = False
#             gesture_start_time = None
#
#         # ☝ Next slide
#         elif fingers == [0, 1, 0, 0, 0]:
#             current_slide = (current_slide + 1) % len(slides)
#             time.sleep(0.4)
#
#         # 👍 Previous slide
#         elif fingers == [1, 0, 0, 0, 0]:
#             current_slide = (current_slide - 1) % len(slides)
#             time.sleep(0.4)
#
#         # ✌ Draw (index + middle)
#         elif fingers == [0, 1, 1, 0, 0]:
#             drawing = True
#
#         # ✌ + 👍 Change color (hold >1 sec)
#         elif fingers == [1, 1, 1, 0, 0]:
#             if gesture_start_time is None:
#                 gesture_start_time = now
#             elif now - gesture_start_time > 1:
#                 color_index = (color_index + 1) % len(colors)
#                 color = colors[color_index]
#                 gesture_start_time = None
#         else:
#             drawing = False
#             last_point = None
#             gesture_start_time = None
#
#         # 🧽 Eraser (index + pinky)
#         if fingers == [0, 1, 0, 0, 1]:
#             x = int(lm[8].x * SLIDE_W)
#             y = int(lm[8].y * SLIDE_H)
#             cv2.circle(canvas, (x, y), 25, (0, 0, 0), -1)
#
#         # ---- drawing on canvas ----
#         if drawing:
#             x = int(lm[8].x * SLIDE_W)
#             y = int(lm[8].y * SLIDE_H)
#             if last_point:
#                 cv2.line(canvas, last_point, (x, y), color, 5)
#             last_point = (x, y)
#
#     # ================== MERGE SLIDE + CANVAS ==================
#     slide = slides[current_slide].copy()
#
#     gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
#     _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#     slide[mask > 0] = canvas[mask > 0]
#
#     # ================== WINDOWS ==================
#     cv2.imshow("Presentation", slide)
#     cv2.imshow("Canvas", canvas)
#     cv2.imshow("Camera", cv2.resize(frame, (600, 400)))
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # ================== CLEANUP ==================
# cap.release()
# cv2.destroyAllWindows()
#












import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Налаштування розмірів (зменшені для зручності) ---
W, H = 680, 420

# --- Ініціалізація MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# --- Кольори ---
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: Червоний, Зелений, Синій
color_index = 0

# --- Завантаження слайдів ---
slides_path = os.path.expanduser("~/Desktop/slides")
slide_files = sorted([f for f in os.listdir(slides_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if not slide_files:
    # Якщо папки немає, створюємо кольорові фони як заглушки
    slides = [np.full((H, W, 3), (50, 50, 50), np.uint8) for _ in range(3)]
else:
    slides = [cv2.resize(cv2.imread(os.path.join(slides_path, f)), (W, H)) for f in slide_files]

# --- Стан застосунку ---
current_slide = 0
canvas = np.zeros((H, W, 3), dtype=np.uint8)
last_point = None
gesture_timer = 0
nav_time = 0

cap = cv2.VideoCapture(0)

print("Запущено! Вікна: Presentation, Canvas, Camera.")

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

            # Визначаємо пальці
            if lm[4].x > lm[3].x + 0.01: fingers[0] = 1  # Великий
            for i, tip in enumerate([8, 12, 16, 20]):
                if lm[tip].y < lm[tip - 2].y: fingers[i + 1] = 1  # Інші

            ix, iy = int(lm[8].x * W), int(lm[8].y * H)

            # 1. Долоня - Очистити і скинути на початок
            if sum(fingers) == 5:
                canvas = np.zeros_like(canvas)
                current_slide = 0

            # 2. Вказівний + Середній - Малювання
            elif fingers[1] and fingers[2] and not fingers[0]:
                if last_point is not None:
                    cv2.line(canvas, last_point, (ix, iy), colors[color_index], 8)
                last_point = (ix, iy)
                gesture_timer = 0

            # 3. Вказівний + Середній + Великий - Зміна кольору (затримка 1с)
            elif fingers[0] and fingers[1] and fingers[2]:
                if gesture_timer == 0:
                    gesture_timer = curr_time
                elif curr_time - gesture_timer > 1.0:
                    color_index = (color_index + 1) % len(colors)
                    gesture_timer = curr_time
                    print("Колір змінено!")
                last_point = None

            # 4. Вказівний (один) - Наступний слайд
            elif fingers[1] and sum(fingers) == 1:
                if curr_time - nav_time > 0.8:
                    current_slide = (current_slide + 1) % len(slides)
                    nav_time = curr_time
                last_point = None

            # 5. Великий (один) - Попередній слайд
            elif fingers[0] and sum(fingers) == 1:
                if curr_time - nav_time > 0.8:
                    current_slide = (current_slide - 1) % len(slides)
                    nav_time = curr_time
                last_point = None

            # 6. Вказівний + Мізинець - Гумка
            elif fingers[1] and fingers[4]:
                cv2.circle(canvas, (ix, iy), 30, (0, 0, 0), -1)
                last_point = None

            else:
                last_point = None
                gesture_timer = 0

    # --- Складання фінального кадру презентації ---
    slide_img = slides[current_slide].copy()
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
    slide_img[mask > 0] = canvas[mask > 0]

    # Візуальний індикатор кольору на слайді
    cv2.circle(slide_img, (30, 30), 15, colors[color_index], -1)

    # --- Вивід трьох вікон ---
    cv2.imshow("Presentation", slide_img)  # 1. Результат
    cv2.imshow("Canvas", canvas)  # 2. Полотно (чорне)
    cv2.imshow("Camera", frame_draw)  # 3. Камера

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


