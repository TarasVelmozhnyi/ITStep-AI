# Завдання 1
# Відкрийте відео з файлу data\lesson7\meter.mp4.
# Проведіть бінарізацію кадрів та збережіть в новий файл.
# Можливо очистіть від шуму або наведіть різкість через
# bilateralFilter


import cv2

video = cv2.VideoCapture('data/lesson7/meter.mp4')

fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


result = cv2.VideoWriter('data/lesson7/meter_binary.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps,
                         (width, height),
                         isColor=False)

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    binary = cv2.adaptiveThreshold(filtered, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   11, 2)

    result.write(binary)

    cv2.imshow('frame', frame)
    cv2.imshow('binary', binary)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
result.release()
