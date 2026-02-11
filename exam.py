# terminal
# cd /Users/tarasvelmozhnyi/Documents/
# python3 hand_presentation.py





import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ============== КОНСТАНТИ ==============
WINDOW_WIDTH, WINDOW_HEIGHT = 680, 420
CAMERA_WIDTH, CAMERA_HEIGHT = 680, 420

# Константи для детекції жестів
THUMB_THRESHOLD = 0.01  # Поріг для детекції великого пальця (відносна координата)
ERASER_RADIUS = 30  # Радіус гумки
LINE_THICKNESS = 8  # Товщина лінії при малюванні
COLOR_CHANGE_DELAY = 1.0  # Затримка зміни кольору (секунди)
NAVIGATION_DELAY = 0.8  # Затримка навігації між слайдами (секунди)

# Константи для масок
MASK_THRESHOLD = 10  # Поріг для виявлення ненульових пікселів у малюнку
DRAWING_THRESHOLD = 10  # Поріг для визначення "непустих" пікселів

# Кольори для малювання (BGR формат)
COLORS = [(0, 0, 255),  # Червоний
          (0, 255, 0),  # Зелений
          (255, 0, 0)]  # Синій
DEFAULT_COLOR_INDEX = 0

# ============== ІНІЦІАЛІЗАЦІЯ ВІКОН ==============
cv2.namedWindow("Presentation")
cv2.namedWindow("Canvas")
cv2.namedWindow("Camera")

# ============== ІНІЦІАЛІЗАЦІЯ MEDIAPIPE ==============
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ============== ЗАВАНТАЖЕННЯ СЛАЙДІВ ==============
slides_path = "/Users/tarasvelmozhnyi/Desktop/slides"

# Отримання списку файлів зображень
slide_files = []
valid_extensions = ('.png', '.jpg', '.jpeg')

for file in os.listdir(slides_path):
    name_lower = file.lower()
    if name_lower.endswith(valid_extensions):
        slide_files.append(file)

# Сортування за назвою
slide_files.sort()

# Завантаження та ресайз слайдів
slides = []
for filename in slide_files:
    img_path = os.path.join(slides_path, filename)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
        slides.append(img)
    else:
        print(f"Попередження: не вдалося завантажити {filename}")

if not slides:
    print("Помилка: не знайдено жодного слайда!")
    exit()

# ============== СТАН ПРОГРАМИ ==============
current_slide = 0
color_index = DEFAULT_COLOR_INDEX
canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
last_point = None
gesture_timer = 0
nav_timer = 0

# ============== ІНІЦІАЛІЗАЦІЯ КАМЕРИ ==============
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Помилка: не вдалося відкрити камеру!")
    exit()

print("Запускаємо вікна: Presentation, Canvas, Camera.")


def compose_slide(slide_img, drawing_img):
    """
    Об'єднує слайд з малюнком.

    Args:
        slide_img: Зображення слайда
        drawing_img: Зображення малюнка на чорному фоні

    Returns:
        Об'єднане зображення
    """
    # Конвертуємо малюнок у відтінки сірого для створення маски
    drawing_gray = cv2.cvtColor(drawing_img, cv2.COLOR_BGR2GRAY)

    # Створюємо маску: True для пікселів, де є малюнок
    mask = drawing_gray > DRAWING_THRESHOLD

    # Копіюємо слайд
    result = slide_img.copy()

    # Замінюємо пікселі слайда малюнком там, де маска=True
    result[mask] = drawing_img[mask]

    return result


def get_finger_states(landmarks):
    """
    Визначає стан пальців (зігнуті/розігнуті).

    Args:
        landmarks: Точки руки від MediaPipe

    Returns:
        Список з 5 елементів: 1=розігнутий, 0=зігнутий
        [великий, вказівний, середній, безіменний, мізинець]
    """
    fingers = [0, 0, 0, 0, 0]
    lm = landmarks

    # Великий палець: порівнюємо x-координати
    if lm[4].x > lm[3].x + THUMB_THRESHOLD:
        fingers[0] = 1  # Великий розігнутий

    # Решта пальців: порівнюємо y-координати
    finger_tips = [8, 12, 16, 20]  # Вказівний, середній, безіменний, мізинець
    finger_pips = [6, 10, 14, 18]  # Суглоби основи пальців

    for i, (tip, pip) in enumerate(zip(finger_tips, finger_pips)):
        if lm[tip].y < lm[pip].y:  # Кінець пальця вище за основу
            fingers[i + 1] = 1  # Палець розігнутий

    return fingers


# ============== РОЗТАШУВАННЯ ВІКОН ==============
cv2.moveWindow("Presentation", 50, 50)  # Ліворуч зверху
cv2.moveWindow("Canvas", WINDOW_WIDTH + 100, 50)  # Праворуч зверху
cv2.moveWindow("Camera", 50, WINDOW_HEIGHT + 100)  # Ліворуч знизу

# ============== ГОЛОВНИЙ ЦИКЛ ==============
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Попередження: не вдалося отримати кадр з камери")
        break

    # Віддзеркалення та ресайз кадру
    frame = cv2.flip(frame, 1)
    frame_draw = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
    rgb_frame = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)

    # Обробка руки за допомогою MediaPipe
    result = hands.process(rgb_frame)

    # Отримання поточного часу
    current_time = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Візуалізація точок та з'єднань руки
            mp_draw.draw_landmarks(
                frame_draw,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Отримання станів пальців
            fingers = get_finger_states(hand_landmarks.landmark)

            # Координати кінчика вказівного пальця
            index_tip = hand_landmarks.landmark[8]
            index_x = int(index_tip.x * WINDOW_WIDTH)
            index_y = int(index_tip.y * WINDOW_HEIGHT)

            # ----- ЖЕСТИ -----

            # 1. ДОЛОНЯ (усі пальці розігнуті) - Очищення та скидання
            if sum(fingers) == 5:
                canvas = np.zeros_like(canvas)
                current_slide = 0
                print("Очищено полотно та повернуто до першого слайда")

            # 2. ВКАЗІВНИЙ + СЕРЕДНІЙ (без великого) - МАЛЮВАННЯ
            elif fingers[1] and fingers[2] and not fingers[0]:
                if last_point is not None:
                    cv2.line(canvas, last_point, (index_x, index_y),
                             COLORS[color_index], LINE_THICKNESS)
                last_point = (index_x, index_y)
                gesture_timer = 0  # Скидання таймера

            # 3. ВКАЗІВНИЙ + СЕРЕДНІЙ + ВЕЛИКИЙ - ЗМІНА КОЛЬОРУ (з затримкою)
            elif fingers[0] and fingers[1] and fingers[2]:
                if gesture_timer == 0:
                    gesture_timer = current_time
                elif current_time - gesture_timer > COLOR_CHANGE_DELAY:
                    color_index = (color_index + 1) % len(COLORS)
                    gesture_timer = current_time
                    print(f"Колір змінено на {COLORS[color_index]}")
                last_point = None

            # 4. ВКАЗІВНИК (один) - НАСТУПНИЙ СЛАЙД
            elif fingers[1] and sum(fingers) == 1:
                if current_time - nav_timer > NAVIGATION_DELAY:
                    current_slide = (current_slide + 1) % len(slides)
                    nav_timer = current_time
                    print(f"Слайд: {current_slide + 1}/{len(slides)}")
                last_point = None

            # 5. ВЕЛИКИЙ (один) - ПОПЕРЕДНІЙ СЛАЙД
            elif fingers[0] and sum(fingers) == 1:
                if current_time - nav_timer > NAVIGATION_DELAY:
                    current_slide = (current_slide - 1) % len(slides)
                    nav_timer = current_time
                    print(f"Слайд: {current_slide + 1}/{len(slides)}")
                last_point = None

            # 6. ВКАЗІВНИЙ + МІЗИНЕЦЬ - ГУМКА
            elif fingers[1] and fingers[4]:
                cv2.circle(canvas, (index_x, index_y),
                           ERASER_RADIUS, (0, 0, 0), -1)
                last_point = None

            # 7. ІНШІ ЖЕСТИ - СКИДАННЯ
            else:
                last_point = None
                gesture_timer = 0

    else:
        # Якщо рука не виявлена - скидаємо стан
        last_point = None
        gesture_timer = 0

    # ============== ВІДОБРАЖЕННЯ ВІКОН ==============

    # 1. Об'єднаний слайд з малюнком
    combined_slide = compose_slide(slides[current_slide], canvas)
    cv2.imshow("Presentation", combined_slide)

    # 2. Чисте полотно для малювання
    cv2.imshow("Canvas", canvas)

    # 3. Відео з камери з розпізнаванням
    cv2.imshow("Camera", frame_draw)

    # ============== ВИХІД З ПРОГРАМИ ==============
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Завершення роботи програми...")
        break

# ============== ЗАВЕРШЕННЯ РОБОТИ ==============
cap.release()
cv2.destroyAllWindows()
print("Програма завершила роботу.")















# import cv2
# import mediapipe as mp
# import numpy as np
# import os
# import time
#
#
# #   розмір
# W, H = 680, 420
#
# cv2.namedWindow("Presentation")
# cv2.namedWindow("Canvas")
# cv2.namedWindow("Camera")
#
# # MediaPipe
#
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     max_num_hands=1,
#     min_detection_confidence=0.7
# )
# mp_draw = mp.solutions.drawing_utils
#
# #  кольори  # BGR
# colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
# color_index = 0
#
# # # слайди
#
# slides_path = "/Users/tarasvelmozhnyi/Desktop/slides"
#
# # файли отримати
# slide_files = []
# for file in os.listdir(slides_path):
#     name_lower = file.lower()
#     if name_lower.endswith(('.png', '.jpg', '.jpeg')):
#         slide_files.append(file)
#
# #  сортування (працює, якщо назви одн
#
# slide_files.sort()
#
# slides = []
# for filename in slide_files:
#     img = cv2.imread(os.path.join(slides_path, filename))
#     img = cv2.resize(img, (W, H))
#     slides.append(img)
#
# # - Стан застосунку ---
#
# current_slide = 0
# canvas = np.zeros((H, W, 3), dtype=np.uint8)
# last_point = None
# gesture_timer = 0
# nav_time = 0
#
# cap = cv2.VideoCapture(0)
#
# print("Запускаємо : Presentation, Canvas, Camera.")
#
#
# def compose_slide(slide, drawing):
#     """Об'єднує слайд з малюнком"""
#
#     result = slide.copy()
#     mask = np.any(drawing > 10, axis=2)
#     result[mask] = drawing[mask]
#
#     return result
#
#
# # Розташування вікон
# cv2.moveWindow("Presentation", 50, 50)  # ліворуч зверху
# cv2.moveWindow("Canvas", W + 100, 50)  # праворуч зверху
# cv2.moveWindow("Camera", 50, H + 100)  # ліворуч знизу
#
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success: break
#
#     frame = cv2.flip(frame, 1)
#     frame_draw = cv2.resize(frame, (W, H))
#     rgb_frame = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb_frame)
#
#     fingers = [0, 0, 0, 0, 0]
#     curr_time = time.time()
#
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame_draw, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             lm = hand_landmarks.landmark
#
#             # Визначаємо які пальці
#             if lm[4].x > lm[3].x + 0.01: fingers[0] = 1  # Великий
#             for i, tip in enumerate([8, 12, 16, 20]):
#                 if lm[tip].y < lm[tip - 2].y: fingers[i + 1] = 1  # Інші
#
#             ix, iy = int(lm[8].x * W), int(lm[8].y * H)
#
#             #  Долоня - Очистити і скинути на початок
#             if sum(fingers) == 5:
#                 canvas = np.zeros_like(canvas)
#                 current_slide = 0
#
#             #  Вказівний + Середній - Малювання
#             elif fingers[1] and fingers[2] and not fingers[0]:
#                 if last_point is not None:
#                     cv2.line(canvas, last_point, (ix, iy), colors[color_index], 8)
#                 last_point = (ix, iy)
#                 gesture_timer = 0
#
#             #  Вказівний + Середній + Великий - Зміна кольору (затримка 1с)
#             elif fingers[0] and fingers[1] and fingers[2]:
#                 if gesture_timer == 0:
#                     gesture_timer = curr_time
#                 elif curr_time - gesture_timer > 1.0:
#                     color_index = (color_index + 1) % len(colors)
#                     gesture_timer = curr_time
#                     print("Колір змінено!")
#                 last_point = None
#
#             #  Вказівний (один) - Наступний слайд
#             elif fingers[1] and sum(fingers) == 1:
#                 if curr_time - nav_time > 0.8:
#                     current_slide = (current_slide + 1) % len(slides)
#                     nav_time = curr_time
#                 last_point = None
#
#             #  Великий (один) - Попередній слайд
#             elif fingers[0] and sum(fingers) == 1:
#                 if curr_time - nav_time > 0.8:
#                     current_slide = (current_slide - 1) % len(slides)
#                     nav_time = curr_time
#                 last_point = None
#
#             #  Вказівний + Мізинець - Гумка
#             elif fingers[1] and fingers[4]:
#                 cv2.circle(canvas, (ix, iy), 30, (0, 0, 0), -1)
#                 last_point = None
#
#             else:
#                 last_point = None
#                 gesture_timer = 0
#
#     slide_img = compose_slide(slides[current_slide], canvas)
#
#     cv2.imshow("Presentation", slide_img)  # 1. Результат
#     cv2.imshow("Canvas", canvas)  # 2. Полотно (чорне)
#     cv2.imshow("Camera", frame_draw)  # 3. Камера
#
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()





import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ============== КОНСТАНТИ ==============
WINDOW_WIDTH, WINDOW_HEIGHT = 680, 420
CAMERA_WIDTH, CAMERA_HEIGHT = 680, 420

# Константи для детекції жестів
THUMB_THRESHOLD = 0.01  # Поріг для детекції великого пальця (відносна координата)
ERASER_RADIUS = 30  # Радіус гумки
LINE_THICKNESS = 8  # Товщина лінії при малюванні
COLOR_CHANGE_DELAY = 1.0  # Затримка зміни кольору (секунди)
NAVIGATION_DELAY = 0.8  # Затримка навігації між слайдами (секунди)

# Константи для масок
MASK_THRESHOLD = 10  # Поріг для виявлення ненульових пікселів у малюнку
DRAWING_THRESHOLD = 10  # Поріг для визначення "непустих" пікселів

# Кольори для малювання (BGR формат)
COLORS = [(0, 0, 255),  # Червоний
          (0, 255, 0),  # Зелений
          (255, 0, 0)]  # Синій
DEFAULT_COLOR_INDEX = 0

# ============== ІНІЦІАЛІЗАЦІЯ ВІКОН ==============
cv2.namedWindow("Presentation")
cv2.namedWindow("Canvas")
cv2.namedWindow("Camera")

# ============== ІНІЦІАЛІЗАЦІЯ MEDIAPIPE ==============
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ============== ЗАВАНТАЖЕННЯ СЛАЙДІВ ==============
slides_path = "/Users/tarasvelmozhnyi/Desktop/slides"

# Отримання списку файлів зображень
slide_files = []
valid_extensions = ('.png', '.jpg', '.jpeg')

for file in os.listdir(slides_path):
    name_lower = file.lower()
    if name_lower.endswith(valid_extensions):
        slide_files.append(file)

# Сортування за назвою
slide_files.sort()

# Завантаження та ресайз слайдів
slides = []
for filename in slide_files:
    img_path = os.path.join(slides_path, filename)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
        slides.append(img)
    else:
        print(f"Попередження: не вдалося завантажити {filename}")

if not slides:
    print("Помилка: не знайдено жодного слайда!")
    exit()

# ============== СТАН ПРОГРАМИ ==============
current_slide = 0
color_index = DEFAULT_COLOR_INDEX
canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
last_point = None
gesture_timer = 0
nav_timer = 0

# ============== ІНІЦІАЛІЗАЦІЯ КАМЕРИ ==============
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Помилка: не вдалося відкрити камеру!")
    exit()

print("Запускаємо вікна: Presentation, Canvas, Camera.")


def compose_slide(slide_img, drawing_img):
    """
    Об'єднує слайд з малюнком.

    Args:
        slide_img: Зображення слайда
        drawing_img: Зображення малюнка на чорному фоні

    Returns:
        Об'єднане зображення
    """
    # Конвертуємо малюнок у відтінки сірого для створення маски
    drawing_gray = cv2.cvtColor(drawing_img, cv2.COLOR_BGR2GRAY)

    # Створюємо маску: True для пікселів, де є малюнок
    mask = drawing_gray > DRAWING_THRESHOLD

    # Копіюємо слайд
    result = slide_img.copy()

    # Замінюємо пікселі слайда малюнком там, де маска=True
    result[mask] = drawing_img[mask]

    return result


def get_finger_states(landmarks):
    """
    Визначає стан пальців (зігнуті/розігнуті).

    Args:
        landmarks: Точки руки від MediaPipe

    Returns:
        Список з 5 елементів: 1=розігнутий, 0=зігнутий
        [великий, вказівний, середній, безіменний, мізинець]
    """
    fingers = [0, 0, 0, 0, 0]
    lm = landmarks

    # Великий палець: порівнюємо x-координати
    if lm[4].x > lm[3].x + THUMB_THRESHOLD:
        fingers[0] = 1  # Великий розігнутий

    # Решта пальців: порівнюємо y-координати
    finger_tips = [8, 12, 16, 20]  # Вказівний, середній, безіменний, мізинець
    finger_pips = [6, 10, 14, 18]  # Суглоби основи пальців

    for i, (tip, pip) in enumerate(zip(finger_tips, finger_pips)):
        if lm[tip].y < lm[pip].y:  # Кінець пальця вище за основу
            fingers[i + 1] = 1  # Палець розігнутий

    return fingers


# ============== РОЗТАШУВАННЯ ВІКОН ==============
cv2.moveWindow("Presentation", 50, 50)  # Ліворуч зверху
cv2.moveWindow("Canvas", WINDOW_WIDTH + 100, 50)  # Праворуч зверху
cv2.moveWindow("Camera", 50, WINDOW_HEIGHT + 100)  # Ліворуч знизу

# ============== ГОЛОВНИЙ ЦИКЛ ==============
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Попередження: не вдалося отримати кадр з камери")
        break

    # Віддзеркалення та ресайз кадру
    frame = cv2.flip(frame, 1)
    frame_draw = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
    rgb_frame = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)

    # Обробка руки за допомогою MediaPipe
    result = hands.process(rgb_frame)

    # Отримання поточного часу
    current_time = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Візуалізація точок та з'єднань руки
            mp_draw.draw_landmarks(
                frame_draw,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Отримання станів пальців
            fingers = get_finger_states(hand_landmarks.landmark)

            # Координати кінчика вказівного пальця
            index_tip = hand_landmarks.landmark[8]
            index_x = int(index_tip.x * WINDOW_WIDTH)
            index_y = int(index_tip.y * WINDOW_HEIGHT)

            # ----- ЖЕСТИ -----

            # 1. ДОЛОНЯ (усі пальці розігнуті) - Очищення та скидання
            if sum(fingers) == 5:
                canvas = np.zeros_like(canvas)
                current_slide = 0
                print("Очищено полотно та повернуто до першого слайда")

            # 2. ВКАЗІВНИЙ + СЕРЕДНІЙ (без великого) - МАЛЮВАННЯ
            elif fingers[1] and fingers[2] and not fingers[0]:
                if last_point is not None:
                    cv2.line(canvas, last_point, (index_x, index_y),
                             COLORS[color_index], LINE_THICKNESS)
                last_point = (index_x, index_y)
                gesture_timer = 0  # Скидання таймера

            # 3. ВКАЗІВНИЙ + СЕРЕДНІЙ + ВЕЛИКИЙ - ЗМІНА КОЛЬОРУ (з затримкою)
            elif fingers[0] and fingers[1] and fingers[2]:
                if gesture_timer == 0:
                    gesture_timer = current_time
                elif current_time - gesture_timer > COLOR_CHANGE_DELAY:
                    color_index = (color_index + 1) % len(COLORS)
                    gesture_timer = current_time
                    print(f"Колір змінено на {COLORS[color_index]}")
                last_point = None

            # 4. ВКАЗІВНИК (один) - НАСТУПНИЙ СЛАЙД
            elif fingers[1] and sum(fingers) == 1:
                if current_time - nav_timer > NAVIGATION_DELAY:
                    current_slide = (current_slide + 1) % len(slides)
                    nav_timer = current_time
                    print(f"Слайд: {current_slide + 1}/{len(slides)}")
                last_point = None

            # 5. ВЕЛИКИЙ (один) - ПОПЕРЕДНІЙ СЛАЙД
            elif fingers[0] and sum(fingers) == 1:
                if current_time - nav_timer > NAVIGATION_DELAY:
                    current_slide = (current_slide - 1) % len(slides)
                    nav_timer = current_time
                    print(f"Слайд: {current_slide + 1}/{len(slides)}")
                last_point = None

            # 6. ВКАЗІВНИЙ + МІЗИНЕЦЬ - ГУМКА
            elif fingers[1] and fingers[4]:
                cv2.circle(canvas, (index_x, index_y),
                           ERASER_RADIUS, (0, 0, 0), -1)
                last_point = None

            # 7. ІНШІ ЖЕСТИ - СКИДАННЯ
            else:
                last_point = None
                gesture_timer = 0

    else:
        # Якщо рука не виявлена - скидаємо стан
        last_point = None
        gesture_timer = 0

    # ============== ВІДОБРАЖЕННЯ ВІКОН ==============

    # 1. Об'єднаний слайд з малюнком
    combined_slide = compose_slide(slides[current_slide], canvas)
    cv2.imshow("Presentation", combined_slide)

    # 2. Чисте полотно для малювання
    cv2.imshow("Canvas", canvas)

    # 3. Відео з камери з розпізнаванням
    cv2.imshow("Camera", frame_draw)

    # ============== ВИХІД З ПРОГРАМИ ==============
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Завершення роботи програми...")
        break

# ============== ЗАВЕРШЕННЯ РОБОТИ ==============
cap.release()
cv2.destroyAllWindows()
print("Програма завершила роботу.")