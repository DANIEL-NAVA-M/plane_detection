from inference_sdk import InferenceHTTPClient

# Conexión al cliente de Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="1x2wL9Kn7iyFRSwLbAdi"
)

# Llamada al modelo usando una imagen local
result = CLIENT.infer("plane.jpg", model_id="airplane-detection-fmc08/2")

# Mostrar los resultados en consola
print(result)
