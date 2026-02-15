import cv2
import mediapipe as mp
import numpy as np
import time
import os


class GestureControlledPresentation:
    def __init__(self, presentation_path, canvas_color=(0, 0, 0)):
        """
        Ініціалізація системи керування презентацією жестами

        Args:
            presentation_path: Шлях до папки з кадрами презентації
            canvas_color: Колір полотна для малювання
        """
        # Ініціалізація MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Шлях до презентації
        self.presentation_path = presentation_path

        # Завантаження кадрів презентації
        self.load_presentation_frames()

        # Поточний слайд
        self.current_slide = 0
        self.total_slides = len(self.presentation_frames)

        # Стан малювання
        self.drawing_mode = False
        self.drawing_color_index = 0
        self.drawing_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Червоний, зелений, синій
        self.drawing_color = self.drawing_colors[self.drawing_color_index]
        self.eraser_mode = False

        # Полотно для малювання
        self.canvas = None
        self.reset_canvas()

        # Траєкторія для малювання
        self.last_point = None

        # Таймери для стабільності жестів
        self.gesture_start_time = {}
        self.gesture_min_duration = 1.0  # Мінімальна тривалість жесту в секундах

        # Стан показу
        self.presentation_active = False

        # Текст для інтерфейсу
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2

        print(f"Завантажено {self.total_slides} слайдів")
        print("Інструкції:")
        print("1. Відкрита долоня - Почати/очистити")
        print("2. Один палець - Наступний слайд")
        print("3. Великий палець вбік - Попередній слайд")
        print("4. Два пальці - Малювання")
        print("5. Два пальці + великий вбік - Зміна кольору")
        print("6. Гумка: жест 'OK' (великий + вказівний палець)")

    def load_presentation_frames(self):
        """Завантаження кадрів презентації з папки"""
        self.presentation_frames = []

        # Перевіряємо, чи існує папка
        if not os.path.exists(self.presentation_path):
            print(f"Папка {self.presentation_path} не знайдена!")
            # Створюємо тестові слайди
            self.create_test_slides()
        else:
            # Завантажуємо всі зображення з папки
            image_files = sorted([f for f in os.listdir(self.presentation_path)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            for img_file in image_files:
                img_path = os.path.join(self.presentation_path, img_file)
                frame = cv2.imread(img_path)
                if frame is not None:
                    self.presentation_frames.append(frame)

            if len(self.presentation_frames) == 0:
                print("Не знайдено зображень у папці! Створюю тестові слайди...")
                self.create_test_slides()

    def create_test_slides(self):
        """Створення тестових слайдів, якщо папка порожня"""
        slide_size = (800, 600)

        for i in range(5):
            # Створюємо слайд з різним фоном і текстом
            if i % 3 == 0:
                bg_color = (255, 255, 255)  # Білий
                text_color = (0, 0, 0)  # Чорний
            elif i % 3 == 1:
                bg_color = (240, 240, 255)  # Світло-синій
                text_color = (0, 0, 100)  # Темно-синій
            else:
                bg_color = (255, 240, 240)  # Світло-червоний
                text_color = (100, 0, 0)  # Темно-червоний

            slide = np.full((slide_size[1], slide_size[0], 3), bg_color, dtype=np.uint8)

            # Додаємо текст
            cv2.putText(slide, f"Слайд {i + 1}", (100, 100),
                        self.font, 2, text_color, 3)
            cv2.putText(slide, "Презентація з керуванням жестами", (100, 200),
                        self.font, 1, text_color, 2)
            cv2.putText(slide, "Відкрита долоня: Почати/очистити", (100, 300),
                        self.font, 0.7, text_color, 1)
            cv2.putText(slide, "Один палець: Наступний слайд", (100, 350),
                        self.font, 0.7, text_color, 1)
            cv2.putText(slide, "Великий палець: Попередній слайд", (100, 400),
                        self.font, 0.7, text_color, 1)
            cv2.putText(slide, "Два пальці: Малювання", (100, 450),
                        self.font, 0.7, text_color, 1)

            self.presentation_frames.append(slide)

        print(f"Створено {len(self.presentation_frames)} тестових слайдів")

    def reset_canvas(self):
        """Скидання полотна для малювання"""
        if len(self.presentation_frames) > 0:
            slide_shape = self.presentation_frames[0].shape
            self.canvas = np.zeros((slide_shape[0], slide_shape[1], 3), dtype=np.uint8)
        else:
            self.canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        self.last_point = None

    def detect_gesture(self, hand_landmarks):
        """
        Визначення жесту на основі ключових точок руки

        Args:
            hand_landmarks: Ключові точки руки від MediaPipe

        Returns:
            Назва жесту
        """
        # Отримуємо координати важливих точок
        landmarks = hand_landmarks.landmark

        # Координати кінчиків пальців
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]

        # Координати основ пальців
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = landmarks[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = landmarks[self.mp_hands.HandLandmark.PINKY_MCP]

        # Визначаємо, чи палець піднятий (кінчик вище за основу)
        thumb_up = thumb_tip.y < thumb_mcp.y
        index_up = index_tip.y < index_mcp.y
        middle_up = middle_tip.y < middle_mcp.y
        ring_up = ring_tip.y < ring_mcp.y
        pinky_up = pinky_tip.y < pinky_mcp.y

        # Визначаємо положення великого пальця відносно вказівного
        thumb_to_index = thumb_tip.x > index_tip.x

        # Жест "ОК" (великий + вказівний палець)
        thumb_index_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
        ok_gesture = thumb_index_distance < 0.05 and not middle_up and not ring_up and not pinky_up

        # Підраховуємо кількість піднятих пальців
        fingers_up = [index_up, middle_up, ring_up, pinky_up]
        num_fingers_up = sum(fingers_up)

        # Визначаємо жест
        if ok_gesture:
            return "ERASER"
        elif num_fingers_up == 5:  # Всі пальці підняті
            return "OPEN_PALM"
        elif num_fingers_up == 1 and index_up:  # Один вказівний палець
            return "ONE_FINGER"
        elif num_fingers_up == 0 and thumb_to_index:  # Великий палець вбік
            return "THUMB_SIDE"
        elif num_fingers_up == 2 and index_up and middle_up:  # Два пальці
            if thumb_to_index:  # Великий палець вбік
                return "TWO_FINGERS_THUMB"
            else:
                return "TWO_FINGERS"

        return "UNKNOWN"

    def handle_gesture(self, gesture, finger_position=None):
        """
        Обробка жестів та виконання відповідних дій

        Args:
            gesture: Назва жесту
            finger_position: Положення пальців для малювання
        """
        current_time = time.time()
        gesture_key = gesture

        # Перевіряємо тривалість жесту для стабільності
        if gesture not in self.gesture_start_time:
            self.gesture_start_time[gesture] = current_time
            return

        gesture_duration = current_time - self.gesture_start_time[gesture]

        # Для більшості жестів потрібна мінімальна тривалість
        if gesture_duration < 0.2:  # Малий дед-тайм для уникнення шуму
            return

        # Жести, що потребують тривалість 1 секунду
        if gesture in ["TWO_FINGERS_THUMB"]:
            if gesture_duration < self.gesture_min_duration:
                return

        # Обробка жестів
        if gesture == "OPEN_PALM":
            self.presentation_active = True
            self.reset_canvas()
            self.drawing_mode = False
            self.eraser_mode = False
            print("Презентацію розпочато. Полотно очищено.")
            self.gesture_start_time.pop(gesture, None)

        elif gesture == "ONE_FINGER" and self.presentation_active:
            if gesture_duration > 0.5:  # Менша затримка для перемикання слайдів
                self.current_slide = min(self.current_slide + 1, self.total_slides - 1)
                print(f"Наступний слайд: {self.current_slide + 1}/{self.total_slides}")
                self.gesture_start_time.pop(gesture, None)

        elif gesture == "THUMB_SIDE" and self.presentation_active:
            if gesture_duration > 0.5:  # Менша затримка для перемикання слайдів
                self.current_slide = max(self.current_slide - 1, 0)
                print(f"Попередній слайд: {self.current_slide + 1}/{self.total_slides}")
                self.gesture_start_time.pop(gesture, None)

        elif gesture == "TWO_FINGERS":
            self.drawing_mode = True
            self.eraser_mode = False
            if finger_position:
                self.draw_on_canvas(finger_position)

        elif gesture == "TWO_FINGERS_THUMB" and self.presentation_active:
            # Зміна кольору малювання
            self.drawing_color_index = (self.drawing_color_index + 1) % len(self.drawing_colors)
            self.drawing_color = self.drawing_colors[self.drawing_color_index]
            color_names = ["ЧЕРВОНИЙ", "ЗЕЛЕНИЙ", "СИНІЙ"]
            print(f"Колір змінено на: {color_names[self.drawing_color_index]}")
            self.gesture_start_time.pop(gesture, None)

        elif gesture == "ERASER" and self.presentation_active:
            self.eraser_mode = True
            self.drawing_mode = False
            if finger_position:
                self.erase_on_canvas(finger_position)

    def draw_on_canvas(self, position):
        """
        Малювання на полотні

        Args:
            position: Положення пальця для малювання
        """
        if position is None:
            return

        # Перетворюємо координати на розміри полотна
        h, w = self.canvas.shape[:2]
        x, y = int(position[0] * w), int(position[1] * h)
        current_point = (x, y)

        # Малюємо лінію від останньої точки до поточної
        if self.last_point is not None:
            if self.eraser_mode:
                # Використовуємо гумку
                cv2.line(self.canvas, self.last_point, current_point, (0, 0, 0), 20)
            else:
                # Малюємо лінію обраним кольором
                cv2.line(self.canvas, self.last_point, current_point, self.drawing_color, 5)

        # Малюємо коло в поточній позиції
        if self.eraser_mode:
            cv2.circle(self.canvas, current_point, 10, (0, 0, 0), -1)
        else:
            cv2.circle(self.canvas, current_point, 5, self.drawing_color, -1)

        self.last_point = current_point

    def erase_on_canvas(self, position):
        """
        Видалення з полотна (гумка)

        Args:
            position: Положення пальця для видалення
        """
        if position is None:
            return

        # Перетворюємо координати на розміри полотна
        h, w = self.canvas.shape[:2]
        x, y = int(position[0] * w), int(position[1] * h)
        current_point = (x, y)

        # Видаляємо (малюємо чорним кольором)
        if self.last_point is not None:
            cv2.line(self.canvas, self.last_point, current_point, (0, 0, 0), 20)

        cv2.circle(self.canvas, current_point, 10, (0, 0, 0), -1)
        self.last_point = current_point

    def get_finger_position(self, hand_landmarks):
        """
        Отримання положення вказівного пальця для малювання

        Args:
            hand_landmarks: Ключові точки руки

        Returns:
            Координати (x, y) нормалізовані до [0, 1]
        """
        landmarks = hand_landmarks.landmark
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        return (index_tip.x, index_tip.y)

    def apply_canvas_to_slide(self, slide):
        """
        Застосування малюнка з полотна на слайд

        Args:
            slide: Поточний слайд презентації

        Returns:
            Слайд з накладеним малюнком
        """
        if slide.shape != self.canvas.shape:
            self.canvas = cv2.resize(self.canvas, (slide.shape[1], slide.shape[0]))

        # Створюємо маску нечорних пікселів
        mask = np.any(self.canvas != [0, 0, 0], axis=-1)

        # Копіюємо слайд
        result = slide.copy()

        # Замінюємо пікселі слайда на пікселі з полотна
        result[mask] = self.canvas[mask]

        return result

    def draw_interface(self, frame, gesture=None):
        """
        Малювання інтерфейсу з інформацією

        Args:
            frame: Кадр для малювання
            gesture: Поточний жест
        """
        # Інформація про слайд
        slide_info = f"Слайд: {self.current_slide + 1}/{self.total_slides}"
        cv2.putText(frame, slide_info, (10, 30), self.font,
                    self.font_scale, (255, 255, 255), self.font_thickness)

        # Стан презентації
        status = "АКТИВНА" if self.presentation_active else "НЕАКТИВНА"
        status_color = (0, 255, 0) if self.presentation_active else (0, 0, 255)
        cv2.putText(frame, f"Презентація: {status}", (10, 60), self.font,
                    self.font_scale, status_color, self.font_thickness)

        # Режим малювання
        if self.drawing_mode:
            mode_text = "МАЛЮВАННЯ"
            color_names = ["ЧЕРВОНИЙ", "ЗЕЛЕНИЙ", "СИНІЙ"]
            color_text = f"Колір: {color_names[self.drawing_color_index]}"
            cv2.putText(frame, mode_text, (10, 90), self.font,
                        self.font_scale, self.drawing_color, self.font_thickness)
            cv2.putText(frame, color_text, (10, 120), self.font,
                        self.font_scale, self.drawing_color, self.font_thickness)

        # Режим гумки
        if self.eraser_mode:
            cv2.putText(frame, "ГУМКА", (10, 90), self.font,
                        self.font_scale, (255, 255, 255), self.font_thickness)

        # Поточний жест
        if gesture:
            cv2.putText(frame, f"Жест: {gesture}", (10, frame.shape[0] - 30), self.font,
                        self.font_scale, (255, 255, 0), self.font_thickness)

        # Інструкції
        instructions = [
            "Відкрита долоня - Почати/очистити",
            "Один палець - Наступний слайд",
            "Великий палець - Попередній слайд",
            "Два пальці - Малювання",
            "Два пальці + великий - Зміна кольору",
            "OK (великий+вказівний) - Гумка"
        ]

        y_offset = frame.shape[0] - 150
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (frame.shape[1] - 400, y_offset + i * 25),
                        self.font, 0.5, (200, 200, 200), 1)

    def run(self):
        """Запуск основного циклу програми"""
        cap = cv2.VideoCapture(0)

        print("\nЗапуск системи керування презентацією...")
        print("Натисніть 'q' для виходу")

        while cap.isOpened():
            # Зчитуємо кадр з камери
            ret, frame = cap.read()
            if not ret:
                print("Не вдалося отримати кадр з камери")
                break

            # Віддзеркалюємо кадр для природності
            frame = cv2.flip(frame, 1)

            # Очищаємо шуми
            frame = cv2.medianBlur(frame, 5)

            # Конвертуємо в RGB для MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Обробка кадру для виявлення руки
            result = self.hands.process(rgb_frame)

            current_gesture = "UNKNOWN"
            finger_position = None

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Визначаємо жест
                    current_gesture = self.detect_gesture(hand_landmarks)

                    # Отримуємо положення пальця для малювання
                    finger_position = self.get_finger_position(hand_landmarks)

                    # Малюємо контури руки
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )

            # Обробляємо жест
            if current_gesture != "UNKNOWN":
                self.handle_gesture(current_gesture, finger_position)

            # Скидаємо точку малювання, якщо жест змінився
            if current_gesture not in ["TWO_FINGERS", "ERASER"]:
                self.last_point = None

            # Відображаємо презентацію, якщо вона активна
            if self.presentation_active and self.current_slide < len(self.presentation_frames):
                # Отримуємо поточний слайд
                slide = self.presentation_frames[self.current_slide].copy()

                # Застосовуємо малюнок з полотна на слайд
                slide_with_drawing = self.apply_canvas_to_slide(slide)

                # Відображаємо слайд
                cv2.imshow('Презентація', slide_with_drawing)

            # Малюємо інтерфейс
            self.draw_interface(frame, current_gesture)

            # Відображаємо кадр з камери
            cv2.imshow('Камера жести', frame)

            # Вихід при натисканні 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Очищення
        cap.release()
        cv2.destroyAllWindows()


# Створення тестової презентації, якщо потрібно
def create_sample_presentation(folder_path="presentation_slides"):
    """Створення тестової презентації"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    slide_size = (800, 600)

    # Створюємо декілька тестових слайдів
    for i in range(1, 6):
        # Створюємо слайд з різним фоном
        if i % 3 == 0:
            bg_color = (255, 255, 200)  # Світло-жовтий
            text_color = (0, 0, 0)
        elif i % 3 == 1:
            bg_color = (200, 255, 200)  # Світло-зелений
            text_color = (0, 0, 0)
        else:
            bg_color = (200, 200, 255)  # Світло-синій
            text_color = (0, 0, 0)

        slide = np.full((slide_size[1], slide_size[0], 3), bg_color, dtype=np.uint8)

        # Додаємо текст
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(slide, f"Слайд {i}", (100, 100), font, 3, text_color, 5)
        cv2.putText(slide, "Презентація з керуванням жестами", (50, 200), font, 1, text_color, 2)
        cv2.putText(slide, "Використовуйте жести для керування", (50, 250), font, 0.8, text_color, 2)

        # Додаємо додатковий контент
        if i == 1:
            cv2.putText(slide, "Відкрита долоня - старт", (50, 350), font, 0.7, text_color, 2)
        elif i == 2:
            cv2.rectangle(slide, (100, 300), (300, 400), (0, 100, 200), -1)
            cv2.circle(slide, (500, 350), 50, (200, 100, 0), -1)
        elif i == 3:
            cv2.line(slide, (100, 300), (700, 300), (0, 0, 0), 5)
            cv2.putText(slide, "Малюйте на слайдах!", (200, 450), font, 1, text_color, 2)
        elif i == 4:
            points = np.array([[200, 250], [400, 150], [600, 250], [500, 400], [300, 400]], np.int32)
            cv2.polylines(slide, [points], True, (0, 0, 200), 3)
        elif i == 5:
            cv2.putText(slide, "Дякую за увагу!", (200, 300), font, 2, text_color, 3)
            cv2.putText(slide, "Натисніть 'q' для виходу", (200, 400), font, 1, text_color, 2)

        # Зберігаємо слайд
        cv2.imwrite(os.path.join(folder_path, f"slide_{i}.jpg"), slide)

    print(f"Створено тестову презентацію у папці '{folder_path}'")


# Основна функція
if __name__ == "__main__":
    # Створюємо тестову презентацію
    presentation_folder = "presentation_slides"
    create_sample_presentation(presentation_folder)

    # Запускаємо систему керування презентацією
    presentation = GestureControlledPresentation(presentation_folder)
    presentation.run()