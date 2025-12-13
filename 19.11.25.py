# Тема: Стеки. Частина 2
# Завдання 1
# Відкрийте зображення data/lesson3/sonet.png. Проведіть
# бінарізацію.
# Обов’язково використайте:
#  розмиття або наведення різкості
#  адаптивну бінарізацію
#  очищеня шумів



import cv2
import numpy as np


img = cv2.imread(r'data/lesson3/sonet.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow("original", img)

kernel_sharp = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

orig_sharp = cv2.filter2D(img,
                       -1,
                       kernel_sharp
                       )

cv2.imshow("orig_sharp", orig_sharp)


blurred_orig = cv2.GaussianBlur(orig_sharp,
                                ksize=(3, 3),
                                sigmaX=5)

cv2.imshow('blurred_orig', blurred_orig)

gauss_blurred = cv2.GaussianBlur(
    img,
    (3, 3),
    3
)
cv2.imshow("Gaussian Blur", gauss_blurred)


adaptive_binary = cv2.adaptiveThreshold(
    img,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    7,
    3
)
cv2.imshow('Adaptive Binary', adaptive_binary)


bilateral_filtered = cv2.bilateralFilter(
    img,
    d=9,
    sigmaColor=75,
    sigmaSpace=75
)
cv2.imshow("Bilateral Filter", bilateral_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()

