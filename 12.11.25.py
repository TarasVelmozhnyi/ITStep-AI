
# Завдання 1
# Відкрийте зображення data/Lenna.png. Прочитайте маски
# data/mask1.png та data/mask2.png.
# Об’єднайте дві маски в одну, скористайтесь cv2.bitwise_or()
# та виведіть результат
# Виведіть ту частину зображення, яка відповідає:
#  mask1
#  mask2
#  mask1 і mask2
# Усі пікселі які не відповідають маскам замінити на 0, перед
# застосуванням змініть тип даних у масці на bool


import cv2
import numpy as np
#
img = cv2.imread('data/Lesson1/Lenna.png')
mask1 = cv2.imread('data/Lesson1/mask1.png', 0)
mask2 = cv2.imread('data/Lesson1/mask2.png', 0)


if img is None:
    print("Помилка: Не вдалося завантажити Lenna.png")
    exit()
if mask1 is None:
    print("Помилка: Не вдалося завантажити mask1.png")
    exit()
if mask2 is None:
    print("Помилка: Не вдалося завантажити mask2.png")
    exit()


combined_mask = cv2.bitwise_or(mask1, mask2)

cv2.imshow('Mask1', mask1)
cv2.imshow('Mask2', mask2)
cv2.imshow('Combined Mask', combined_mask)


mask1_bool = mask1.astype(bool)
mask2_bool = mask2.astype(bool)

#  mask1
result1 = img.copy()
result1[~mask1_bool] = 0
cv2.imshow('Only Mask1', result1)


#  mask2
result2 = img.copy()
result2[~mask2_bool] = 0
cv2.imshow('Only Mask2', result2)

#  mask1 і mask2
both_mask = mask1_bool & mask2_bool
result3 = img.copy()
result3[~both_mask] = 0
cv2.imshow('Mask1 AND Mask2', result3)

#
cv2.waitKey(0)
cv2.destroyAllWindows()



# 2  Виведіть зображення. Підберіть самостійно межі


img = cv2.imread('data//Lesson1//baboo.jpg')

if img is None:
    print('<UNK>')

print(f"розмір зображення: {img.shape}")

cv2.imshow('image', img)

crop = img[100:300, 100:300]
print(f" розмір після обрізання: {crop.shape}")

cv2.imshow('crop', crop)

cv2.waitKey(0)
cv2.destroyAllWindows()



