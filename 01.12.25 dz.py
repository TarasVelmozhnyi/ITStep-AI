# Завдання 1
# Відкрийте зображення data/lesson_seg/tumor1.jpg Проведіть сегментацію зображення використовуючи
# модель data/lesson_seg/brain-tumor-seg.jpg Визначте площу пухлини в пікселях.
# Визначте площу в cm2 (1 піксель – 0,0025 cm2# )
# В залежності від площі присвойте пухлині певний тип
#  <10 – small
#  10-25 – middle
#  >25 – large
# Покажіть пухлину – за допомогою маски усі лишні пікселі зробіть 0, а як назву зображення використайте її тип




import cv2
import numpy as np
import ultralytics


image = cv2.imread('data/lesson_seg/tumor1.jpg')

model = ultralytics.YOLO('data/lesson_seg/brain-tumor-seg.pt')


cv2.imshow('original', image)

results = model.predict(image)
result = results[0]
res_img = result.plot(
    boxes=True,
    masks=True
)

names = result.names
masks = result.masks.data

cls = result.boxes.cls

nums = len(cls)
heigh, width, _ = image.shape
for i in range(nums):
    name = names[int(cls[i])]
    mask = masks[i]
    mask = mask.numpy()
    mask = mask.astype(np.uint8)
    mask *= 255
    mask = cv2.resize(mask, (width, heigh))
    mask_bool = mask.astype(bool)
    tumor_area_pixels = np.sum(mask_bool)
    tumor_area_cm2 = tumor_area_pixels * 0.0025
    print(f'Площа пухлини (cm**2): {tumor_area_cm2}')
    if tumor_area_cm2 < 10:
        tumor_type = 'small'

    elif 10 <= tumor_area_cm2 <= 25:
        tumor_type = 'middle'

    else:
        tumor_type = 'large'

    img_current = image.copy()
    img_current[~mask_bool] = 0
    cv2.imshow(f'Тип пухлини: {tumor_type}', img_current)
    cv2.waitKey(0)

