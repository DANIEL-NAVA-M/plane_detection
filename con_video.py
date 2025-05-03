import cv2
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Cliente de Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="1x2wL9Kn7iyFRSwLbAdi"
)

# Cargar video
cap = cv2.VideoCapture("varios.mp4")  # Cambia el nombre si es necesario

# Fuente para texto
font = ImageFont.load_default()

# Función para crear un nuevo filtro de Kalman
def create_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)  # 4 estados (x, y, dx, dy) y 2 mediciones (x, y)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Ruido de proceso
    return kf

# Crear filtro de Kalman inicial
kalman = create_kalman_filter()

# Para guardar la trayectoria
trajectory = []

# Arbir video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Guardar frame como imagen temporal
    temp_path = "temp_frame.jpg"
    cv2.imwrite(temp_path, frame)

    # Ejecutar inferencia
    result = CLIENT.infer(temp_path, model_id="airplane-detection-fmc08/2")

    # Convertir imagen a PIL para dibujar
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    measurement_updated = False

    # aplicar la preduccion
    for prediction in result["predictions"]:
        # si el confidence es menor a 65% se ignora
        if prediction["confidence"] < 0.65:  # 65%
            continue

        x, y = prediction["x"], prediction["y"]
        w_box, h_box = prediction["width"], prediction["height"]
        left = x - w_box / 2
        top = y - h_box / 2
        right = x + w_box / 2
        bottom = y + h_box / 2

        # Dibuja el bounding box y etiqueta con su confidence
        label = f"{prediction['class']}: {prediction['confidence']:.2f}"
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        draw.text((left, top - 20), label, fill="red", font=font)

        # Actualiza la medición del filtro de Kalman
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        kalman.correct(measurement)
        measurement_updated = True

    # Predicción con Kalman Filter (si no hay medición, igual predice)
    prediction = kalman.predict()
    predicted_x, predicted_y = int(prediction[0]), int(prediction[1])

    # Dibujar el punto predicho
    draw.ellipse((predicted_x - 5, predicted_y - 5, predicted_x + 5, predicted_y + 5), fill="blue")

    # Guardar en trayectoria
    trajectory.append((predicted_x, predicted_y))

    # Dibujar la trayectoria sobre image_pil
    for i in range(1, len(trajectory)):
        draw.line([trajectory[i-1], trajectory[i]], fill="blue", width=2)

    # Convertir de nuevo a formato OpenCV para mostrar
    frame_drawn = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("Detección de Aviones + Kalman", frame_drawn)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        kalman = create_kalman_filter()  # Reinicia el filtro
        trajectory = []                  # Limpia la trayectoria
        print(">>> Kalman y trayectoria reiniciados <<<")

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
if os.path.exists(temp_path):
    os.remove(temp_path)
