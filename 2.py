import cv2
import numpy as np
import mediapipe as mp
import os
import time

# Ініціалізація MediaPipe
mp_hands = mp.solutions.hands
# На M1 краще використовувати model_complexity=0 або 1 для вищого FPS
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- Шляхи для macOS ---
desktop_path = os.path.expanduser("~/Desktop")
slides_folder = os.path.join(desktop_path, "slides")

if not os.path.exists(slides_folder):
    print(f"Помилка: Папка {slides_folder} не знайдена!")
    exit()

# Завантаження слайдів
slides = []
file_list = sorted([f for f in os.listdir(slides_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

for file in file_list:
    img = cv2.imread(os.path.join(slides_folder, file))
    if img is not None:
        # На Mac екрани часто мають високу роздільну здатність (Retina),
        # приведемо слайди до одного розміру для зручності малювання
        img = cv2.resize(img, (1280, 720))
        slides.append(img)
        print(f"Завантажено: {file}")

if not slides:
    print("У папці немає зображень!")
    exit()

# Налаштування
current_slide = 0
drawing = False
color = (0, 0, 255)  # BGR (Червоний)
canvas = np.zeros_like(slides[0])
last_gesture_time = 0
gesture_cooldown = 0.8  # Затримка між жестами в секундах

# Ініціалізація камери
cap = cv2.VideoCapture(0)

print("\n=== УПРАВЛІННЯ (macOS M1) ===")
print("✋ Долоня - Очистити малюнок")
print("☝️ 1 палець - Наступний слайд")
print("👍 Великий палець - Попередній слайд")
print("✌️ 2 пальці - Увімкнути/Вимкнути малювання")
print("Клавіша 'Q' - Вихід, 'C' - Зміна кольору")
print("============================\n")

last_point = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    curr_time = time.time()

    # Обробка жестів
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    fingers_up = [0, 0, 0, 0, 0]

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark

            # Логіка визначення піднятих пальців (для Mac без змін)
            # Великий палець (враховуємо інверсію x через flip)
            if lm[4].x < lm[3].x - 0.02: fingers_up[0] = 1
            # Інші 4 пальці
            tip_ids = [8, 12, 16, 20]
            pip_ids = [6, 10, 14, 18]
            for i, (tip, pip) in enumerate(zip(tip_ids, pip_ids)):
                if lm[tip].y < lm[pip].y:
                    fingers_up[i + 1] = 1

            # --- Обробка команд із захистом від частих спрацювань ---
            if curr_time - last_gesture_time > gesture_cooldown:

                # 1. Очищення (Долоня)
                if sum(fingers_up) == 5:
                    canvas = np.zeros_like(slides[0])
                    last_gesture_time = curr_time
                    print("Очищено")

                # 2. Наступний слайд (Вказівний)
                elif fingers_up == [0, 1, 0, 0, 0]:
                    if current_slide < len(slides) - 1:
                        current_slide += 1
                        canvas = np.zeros_like(slides[0])  # очистити при переході
                        last_gesture_time = curr_time
                        print(f"Слайд: {current_slide + 1}")

                # 3. Попередній слайд (Великий)
                elif fingers_up == [1, 0, 0, 0, 0]:
                    if current_slide > 0:
                        current_slide -= 1
                        canvas = np.zeros_like(slides[0])
                        last_gesture_time = curr_time
                        print(f"Слайд: {current_slide + 1}")

                # 4. Режим малювання (Два пальці)
                elif fingers_up == [0, 1, 1, 0, 0]:
                    drawing = not drawing
                    last_gesture_time = curr_time
                    print(f"Малювання: {drawing}")

            # --- Процес малювання ---
            if drawing and fingers_up[1] == 1:
                # Масштабуємо координати під розмір слайда (1280x720)
                idx_x = int(lm[8].x * 1280)
                idx_y = int(lm[8].y * 720)

                if last_point is not None:
                    cv2.line(canvas, last_point, (idx_x, idx_y), color, 6)
                last_point = (idx_x, idx_y)
            else:
                last_point = None

    # Накладання малюнка на поточний слайд
    current_img = slides[current_slide].copy()
    mask = np.any(canvas != [0, 0, 0], axis=2)
    current_img[mask] = canvas[mask]

    # Вивід вікон
    cv2.imshow('Presentation View', current_img)

    # Індикатор на вікні камери
    status_color = (0, 255, 0) if drawing else (0, 0, 255)
    cv2.putText(frame, f"DRAW: {drawing}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.imshow('Control Camera', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

cap.release()
cv2.destroyAllWindows()




#
#
# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# import time
#
# # --- Налаштування MediaPipe ---
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     model_complexity=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.5
# )
# mp_draw = mp.solutions.drawing_utils
#
# # --- Робота з файлами ---
# desktop_path = os.path.expanduser("~/Desktop")
# slides_folder = os.path.join(desktop_path, "slides")
#
# if not os.path.exists(slides_folder):
#     print(f"Помилка: Папка {slides_folder} не знайдена!")
#     exit()
#
# slides = []
# file_list = sorted([f for f in os.listdir(slides_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
#
# # Розміри для виводу
# SLIDE_W, SLIDE_H = 800, 500  # Зменшений розмір слайда
# CAM_W, CAM_H = 400, 300  # Зменшене вікно камери
#
# for file in file_list:
#     img = cv2.imread(os.path.join(slides_folder, file))
#     if img is not None:
#         img = cv2.resize(img, (SLIDE_W, SLIDE_H))
#         slides.append(img)
#
# if not slides:
#     print("Слайди не знайдено!")
#     exit()
#
# # --- Стан програми ---
# current_slide = 0
# drawing = False
# color = (0, 255, 0)  # Почнемо з зеленого
# canvas = np.zeros((SLIDE_H, SLIDE_W, 3), dtype=np.uint8)
# last_gesture_time = 0
# gesture_cooldown = 0.7
# last_point = None
#
# cap = cv2.VideoCapture(0)
#
# # Створюємо вікна заздалегідь для керування розміром
# cv2.namedWindow('Slide', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Slide', SLIDE_W, SLIDE_H)
#
# cv2.namedWindow('Drawing Board', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Drawing Board', SLIDE_W, SLIDE_H)
#
# cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Camera', CAM_W, CAM_H)
#
# print("=== ЗАПУЩЕНО ===")
# print("✋ Долоня - Очистити")
# print("☝️ 1 палець - Наступний")
# print("👍 Великий - Попередній")
# print("✌️ 2 пальці - Малювання ВКЛ/ВИКЛ")
#
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success: break
#
#     frame = cv2.flip(frame, 1)
#     frame_small = cv2.resize(frame, (CAM_W, CAM_H))
#     curr_time = time.time()
#
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb)
#
#     fingers_up = [0, 0, 0, 0, 0]
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame_small, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             lm = hand_landmarks.landmark
#
#             # Визначення жестів (враховуючи координати MediaPipe 0.0 - 1.0)
#             if lm[4].x < lm[3].x - 0.02: fingers_up[0] = 1  # Великий
#             for i, (tip, pip) in enumerate(zip([8, 12, 16, 20], [6, 10, 14, 18])):
#                 if lm[tip].y < lm[pip].y: fingers_up[i + 1] = 1
#
#             # Команди
#             if curr_time - last_gesture_time > gesture_cooldown:
#                 if sum(fingers_up) == 5:  # Очистити
#                     canvas = np.zeros_like(canvas)
#                     last_gesture_time = curr_time
#                 elif fingers_up == [0, 1, 0, 0, 0]:  # Вперед
#                     current_slide = min(current_slide + 1, len(slides) - 1)
#                     last_gesture_time = curr_time
#                 elif fingers_up == [1, 0, 0, 0, 0]:  # Назад
#                     current_slide = max(current_slide - 1, 0)
#                     last_gesture_time = curr_time
#                 elif fingers_up == [0, 1, 1, 0, 0]:  # Режим малювання
#                     drawing = not drawing
#                     last_gesture_time = curr_time
#
#             # Малювання (використовуємо вказівний палець)
#             if drawing and fingers_up[1] == 1:
#                 # Переводимо відносні координати руки (0-1) у розмір слайда
#                 x = int(lm[8].x * SLIDE_W)
#                 y = int(lm[8].y * SLIDE_H)
#
#                 if last_point is not None:
#                     cv2.line(canvas, last_point, (x, y), color, 5)
#                 last_point = (x, y)
#             else:
#                 last_point = None
#
#     # 1. Формуємо вікно Слайда
#     slide_view = slides[current_slide].copy()
#     mask = np.any(canvas != [0, 0, 0], axis=2)
#     slide_view[mask] = canvas[mask]
#
#     # 2. Виводимо три вікна
#     cv2.imshow('Slide', slide_view)  # Слайд + малюнок
#     cv2.imshow('Drawing Board', canvas)  # Тільки малювання на чорному
#
#     # Додаємо текст стану на камеру
#     status = "DRAWING: ON" if drawing else "DRAWING: OFF"
#     cv2.putText(frame_small, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     cv2.imshow('Camera', frame_small)  # Камера (маленька)
#
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('c'):  # Зміна кольору випадково
#         color = list(np.random.random(size=3) * 256)
#
# cap.release()
# cv2.destroyAllWindows()


#
#
#
#
# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# import time
#
# # --- Ініціалізація MediaPipe ---
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     model_complexity=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.5
# )
# mp_draw = mp.solutions.drawing_utils
#
# # --- Шляхи ---
# desktop_path = os.path.expanduser("~/Desktop")
# slides_folder = os.path.join(desktop_path, "slides")
#
# if not os.path.exists(slides_folder):
#     print(f"Помилка: Папка {slides_folder} не знайдена!")
#     exit()
#
# # Параметри вікон
# SLIDE_W, SLIDE_H = 700, 450
# CANVAS_W, CANVAS_H = 700, 450
# CAM_W, CAM_H = 600, 400
#
# slides = []
# file_list = sorted([f for f in os.listdir(slides_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
#
# for file in file_list:
#     img = cv2.imread(os.path.join(slides_folder, file))
#     if img is not None:
#         slides.append(cv2.resize(img, (SLIDE_W, SLIDE_H)))
#
# if not slides:
#     print("Слайдів немає!")
#     exit()
#
# # --- Стан ---
# current_slide = 0
# drawing = False
# color = (0, 255, 0)  # Зелений за замовчуванням
# canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
# last_gesture_time = 0
# gesture_cooldown = 0.6
# last_point = None
#
# cap = cv2.VideoCapture(0)
#
# # Налаштування вікон macOS
# cv2.namedWindow('Slide', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Slide', SLIDE_W, SLIDE_H)
#
# cv2.namedWindow('Drawing Board', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Drawing Board', CANVAS_W, CANVAS_H)
#
# cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Camera', CAM_W, CAM_H)
#
# print("Система готова!")
#
#
#
#
#
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success: break
#
#     frame = cv2.flip(frame, 1)
#     frame_small = cv2.resize(frame, (CAM_W, CAM_H))
#     curr_time = time.time()
#
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb)
#
#     fingers_up = [0, 0, 0, 0, 0]
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame_small, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             lm = hand_landmarks.landmark
#
#             # Логіка пальців
#             if lm[4].x < lm[3].x - 0.02: fingers_up[0] = 1  # Великий
#             for i, (tip, pip) in enumerate(zip([8, 12, 16, 20], [6, 10, 14, 18])):
#                 if lm[tip].y < lm[pip].y: fingers_up[i + 1] = 1
#
#             # Команди жестів
#             if curr_time - last_gesture_time > gesture_cooldown:
#                 # ✋ Долоня - Очистити малюнок
#                 if sum(fingers_up) == 5:
#                     canvas = np.zeros_like(canvas)
#                     last_gesture_time = curr_time
#
#                 # ☝️ Один вказівний - Наступний слайд
#                 elif fingers_up == [0, 1, 0, 0, 0]:
#                     current_slide = (current_slide + 1) % len(slides)
#                     last_gesture_time = curr_time
#
#                 # 👍 Великий палець - Попередній слайд
#                 elif fingers_up == [1, 0, 0, 0, 0]:
#                     current_slide = (current_slide - 1) % len(slides)
#                     last_gesture_time = curr_time
#
#                 # ✌️ Два пальці - Малювання ВКЛ/ВИКЛ
#                 elif fingers_up == [0, 1, 1, 0, 0]:
#                     drawing = not drawing
#                     last_gesture_time = curr_time
#
#             # Процес малювання (тільки на Canvas)
#             if drawing and fingers_up[1] == 1:
#                 x = int(lm[8].x * CANVAS_W)
#                 y = int(lm[8].y * CANVAS_H)
#
#                 if last_point is not None:
#                     cv2.line(canvas, last_point, (x, y), color, 5)
#                 last_point = (x, y)
#             else:
#                 last_point = None
#
#     # Відображення
#     cv2.imshow('Slide', slides[current_slide])  # Чистий слайд
#     cv2.imshow('Drawing Board', canvas)  # Тільки малювання
#
#     # Статус на камері
#     mode_text = f"DRAW: {'ON' if drawing else 'OFF'}"
#     cv2.putText(frame_small, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#     cv2.imshow('Camera', frame_small)
#
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('c'):  # Змінити колір на випадковий
#         color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
#
# cap.release()
# cv2.destroyAllWindows()