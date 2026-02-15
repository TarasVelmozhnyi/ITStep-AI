import cv2
import mediapipe as mp
import numpy as np
import os
import time
import dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate

# --- Налаштування середовища та моделі ---
dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Створення LLM
llm = GoogleGenerativeAI(
    model='gemini-2.5-flash',  # Використовуємо flash для швидкості
    api_key=gemini_api_key,
)

# Шаблон для генерації контенту
prompt = PromptTemplate.from_template(
    """
    Ти - спеціаліст, який допомагає робити презентації людям. Тобі дають заголовки на слайдах. 
    Ти пишеш підходящий текст до цього заголовку. Відповідай тільки вмістом, без додаткового спілкування.
    Пиши 3-5 речень суцільним текстом без модифікації тексту яким-небудь способом.

    {current_title}
    """
)

chain = prompt | llm

# --- Ініціалізація MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,  # Зменшено для продуктивності
    max_num_hands=2,
)
mp_draw = mp.solutions.drawing_utils

# --- Шляхи ---
desktop_path = os.path.expanduser("~/Desktop")
slides_folder = os.path.join(desktop_path, "slides")


# Перевірка наявності папки
if not os.path.exists(slides_folder):
    print(f"Помилка: Папка '{slides_folder}' не знайдена!")
    exit(1)

# Завантаження слайдів
slide_files = [f for f in os.listdir(slides_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
if not slide_files:
    print(f"Помилка: У папці '{slides_folder}' не знайдено зображень!")
    exit(1)

# Завантаження зображень слайдів
slides = []
for slide_file in slide_files:
    slide_path = os.path.join(slides_folder, slide_file)
    slide_img = cv2.imread(slide_path)
    if slide_img is not None:
        slides.append(slide_img)
    else:
        print(f"Попередження: Не вдалося завантажити {slide_file}")

if not slides:
    print("Помилка: Не вдалося завантажити жодного слайду!")
    exit(1)

# --- Параметри вікон ---
SLIDE_W, SLIDE_H = 600, 400
CANVAS_W, CANVAS_H = 600, 400
CAM_W, CAM_H = 600, 400

# --- Стан ---
current_slide = 0
drawing = False
color = (0, 255, 0)  # Зелений за замовчуванням
canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
last_gesture_time = 0
gesture_cooldown = 0.6
last_point = None
slide_titles = ["Заголовок 1", "Заголовок 2", "Заголовок 3"]  # Можете змінити на реальні заголовки

# Ініціалізація камери
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Помилка: Не вдалося відкрити камеру!")
    exit(1)

# Налаштування вікон
cv2.namedWindow('Slide', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Slide', SLIDE_W, SLIDE_H)

cv2.namedWindow('Drawing Board', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Drawing Board', CANVAS_W, CANVAS_H)

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera', CAM_W, CAM_H)

print("=== Система керування презентацією запущена ===")
print("Жести:")
print("✋ Долоня (всі пальці) - Очистити малюнок")
print("☝️ Вказівний палець - Наступний слайд")
print("👍 Великий палець - Попередній слайд")
print("✌️ Два пальці - Увімкнути/вимкнути малювання")
print("Клавіши:")
print("'q' - Вихід")
print("'c' - Змінити колір малювання")
print("'g' - Генерувати текст для поточного слайду (Gemini)")

# Головний цикл
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Помилка: Не вдалося отримати кадр з камери!")
        break

    frame = cv2.flip(frame, 1)
    frame_small = cv2.resize(frame, (CAM_W, CAM_H))
    curr_time = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    fingers_up = [0, 0, 0, 0, 0]  # [thumb, index, middle, ring, pinky]

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_small, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark

            # Логіка визначення піднятих пальців
            # Великий палець (правильніше для правої руки)
            if lm[4].x < lm[3].x:
                fingers_up[0] = 1

            # Інші пальці
            tip_ids = [8, 12, 16, 20]  # Вказівний, середній, безіменний, мізинець
            pip_ids = [6, 10, 14, 18]

            for i, (tip, pip) in enumerate(zip(tip_ids, pip_ids)):
                if lm[tip].y < lm[pip].y:
                    fingers_up[i + 1] = 1

            # Команди жестів
            if curr_time - last_gesture_time > gesture_cooldown:
                # ✋ Долоня - Очистити малюнок
                if sum(fingers_up) == 5:
                    canvas = np.zeros_like(canvas)
                    last_gesture_time = curr_time
                    print("Очищено полотно для малювання")

                # ☝️ Один вказівний - Наступний слайд
                elif fingers_up == [0, 1, 0, 0, 0]:
                    current_slide = (current_slide + 1) % len(slides)
                    last_gesture_time = curr_time
                    print(f"Слайд {current_slide + 1}/{len(slides)}")

                # 👍 Великий палець - Попередній слайд
                elif fingers_up == [1, 0, 0, 0, 0]:
                    current_slide = (current_slide - 1) % len(slides)
                    last_gesture_time = curr_time
                    print(f"Слайд {current_slide + 1}/{len(slides)}")

                # ✌️ Два пальці - Малювання ВКЛ/ВИКЛ
                elif fingers_up == [0, 1, 1, 0, 0]:
                    drawing = not drawing
                    last_gesture_time = curr_time
                    print(f"Малювання: {'УВІМКНЕНО' if drawing else 'ВИМКНЕНО'}")

            # Процес малювання (тільки на Canvas)
            if drawing and fingers_up[1] == 1:
                x = int(lm[8].x * CANVAS_W)
                y = int(lm[8].y * CANVAS_H)

                if last_point is not None:
                    cv2.line(canvas, last_point, (x, y), color, 5)
                last_point = (x, y)
            else:
                last_point = None

    # Підготовка слайду для відображення
    slide_display = cv2.resize(slides[current_slide], (SLIDE_W, SLIDE_H))

    # Додаємо номер слайду
    cv2.putText(slide_display, f"Слайд {current_slide + 1}/{len(slides)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Відображення всіх вікон
    cv2.imshow('Slide', slide_display)
    cv2.imshow('Drawing Board', canvas)

    # Статус на камері
    mode_text = f"Малювання: {'УВІМК' if drawing else 'ВИМК'}"
    cv2.putText(frame_small, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame_small, f"Слайд: {current_slide + 1}/{len(slides)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.imshow('Camera', frame_small)

    # Обробка клавіш
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  # Змінити колір на випадковий
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        print(f"Новий колір: {color}")
    elif key == ord('g'):  # Генерувати текст через Gemini
        try:
            if current_slide < len(slide_titles):
                title = slide_titles[current_slide]
                print(f"Генерація тексту для: {title}")
                response = chain.invoke({"current_title": title})
                print("=== Згенерований текст ===")
                print(response)
                print("=========================")
            else:
                print("Попередження: Немає заголовка для цього слайду")
        except Exception as e:
            print(f"Помилка генерації тексту: {e}")

# Очищення
cap.release()
cv2.destroyAllWindows()
print("=== Система завершила роботу ===")