


import onnxruntime as ort
from PIL import Image



# відкриваємо модель
session = ort.InferenceSession(
    "model.onnx"
)

print(session)