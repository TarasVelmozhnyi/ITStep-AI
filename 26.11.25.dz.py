
# Тема: Langchain. Частина 3
# Завдання 1
# Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та виведіть результат, підберіть
# параметри
# Можете змінити розмір кадру для кращої візуалізації
# cv2.resize()

import cv2
import ultralytics


model = ultralytics.YOLO('yolov8s.pt')

video = cv2.VideoCapture('data/lesson8/meetings.mp4')

while True:
    success, frame = video.read()
    if not success:
        break

    small_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    results = model.predict(
        small_frame,
        device='cpu',
        conf=0.25,
        iou=0.7,
        classes=[0, 1]
    )

    annotated_frame = results[0].plot()

    cv2.imshow('YOLO Detection', annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


video.release()



#
# Завдання 2
# Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та почніть показувати відео з
# моменту, коли людей стало 5


import cv2
import ultralytics

model = ultralytics.YOLO('yolov8s.pt')
video = cv2.VideoCapture('data/lesson8/meetings.mp4')

show_video = False

while True:
    success, frame = video.read()
    if not success:
        break

    small_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    results = model.predict(
        small_frame,
        device='cpu',
        conf=0.25,
        iou=0.7,
        classes=[0])

    people_count = len(results[0].boxes) if results[0].boxes is not None else 0
    annotated_frame = results[0].plot()

    if people_count > 4:
        show_video = True

    if show_video:
            cv2.imshow('YOLO Detection', annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()