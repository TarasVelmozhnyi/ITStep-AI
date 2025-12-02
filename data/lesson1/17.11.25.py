# Модуль 12. Структури даних
# Тема: Стеки. Частина 2
# Завдання 1
# Відкрийте зображення data\lesson2\darken.png. Проведіть з
# ним наступні операції, переведіть його в HSV формат та
# обробіть канал Value наступними способами:
#  застосуйте вирівнювання гістограм
#  збільшіть значення десь на 20-50%, оскільки тут
# результат буде типу float32 та явно вийде за межі [0-255]
# застосуйте np.clip(value, 0, 255) та value.astype(np.uint8)
# Виведіть результати обох обробок на екран

import cv2
import numpy as np

#  не взяло звідси зображення data\lesson2\darken.png

image = cv2.imread('/Users/tarasvelmozhnyi/Desktop/darken.png')

if image is None:
    print(" не вдалося завантажити зображення")
else:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v_equalized = cv2.equalizeHist(v)
    hsv_equalized = cv2.merge([h, s, v_equalized])
    result1 = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

    v_float = v.astype(np.float32)
    v_increased = v_float * 1.3
    v_increased = np.clip(v_increased, 0, 255)
    v_increased = v_increased.astype(np.uint8)

    hsv_increased = cv2.merge([h, s, v_increased])
    result2 = cv2.cvtColor(hsv_increased, cv2.COLOR_HSV2BGR)

    cv2.imshow('Оригінал', image)
    cv2.imshow('Вирівняна гістограма', result1)
    cv2.imshow('Яскравість +30%', result2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()