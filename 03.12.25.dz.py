# Тема: opencv. Частина 2
# Завдання 1
# Відкрийте відео data/lesson_pose/squat.mp4
# Ваша задача рахувати кількість присідань.
# Отримайте перший кадр та виділіть основні точки.
# Отримайте координати 3-ох точок ноги
# Визначте кут між цими трьома точками. Скористайтесь
# функцією utils.get_angle(x1, y1, x2, y2, x3, y3) де x2, y2 –
# координати коліна(центральна точка)
# Запустіть відео та добавте на сам кадр кут згинання ніг.
# Визначіть нижню межу кута(якщо людина опустилась
# нижче вважаємо що вона достатньо опустилась) та верхню
# межу кута(якщо людина піднялась вище вважаємо що вона
# достатньо піднялась)
# Добавте кількість присідань та
# кут на кожен кадр.


import cv2
import numpy as np
import ultralytics


model = ultralytics.YOLO('yolo11s-pose.pt')
video = cv2.VideoCapture('data/lesson_pose/squat.mp4')

squatting = 0
omitted = False

def get_angle(x1, y1, x2, y2, x3, y3):
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    c = np.array([x3, y3])

    ab = a - b
    cb = c - b

    dot = ab @ cb
    norm_ab = (ab @ ab) ** 0.5
    norm_cb = (cb @ cb) ** 0.5
    angle = np.arccos(dot / norm_ab / norm_cb)
    angle = angle / np.pi * 180

    return angle

while True:
    success, frame = video.read()
    if not success:
        break

    frame = cv2.resize(frame, (640, 480))

    results = model(frame, verbose=False)

    if results and len(results[0].keypoints) > 0:
        keypoints = results[0].keypoints.data[0].cpu().numpy()

        thigh = keypoints[12]
        knee = keypoints[14]
        ankle = keypoints[16]

        a = np.array(thigh[:2])
        b = np.array(knee[:2])
        c = np.array(ankle[:2])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

        if angle < 100 and not omitted:
            omitted = True
        elif angle > 160 and omitted:
            squatting = squatting + 1
            omitted = False

        cv2.putText(frame, f"кут: {int(angle)}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for point in [thigh, knee, ankle]:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 2, (255, 0, 0), 2)

    cv2.putText(frame, f"Присідання: {squatting}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('кількість присідань', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

video.release()
cv2.destroyAllWindows()
print(f"загальна кількість присідань: {squatting}")