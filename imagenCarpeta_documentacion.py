import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt
from skimage.filters import threshold_otsu
import os

# Configuración de la ruta de Tesseract OCR
pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Carga el modelo de detección de objetos una vez fuera del bucle
model = tf.keras.models.load_model('./static/models/object_detection.h5')

# Conjunto de caracteres permitidos para el reconocimiento de texto OCR
KEYS = "0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def object_detection(image):
    """
    Realiza la detección de objetos en una imagen.

    Parameters:
    image (numpy.ndarray): La imagen de entrada en formato NumPy.

    Returns:
    numpy.ndarray: Las coordenadas de los objetos detectados en el formato (xmin, xmax, ymin, ymax).
    """
    # Lee la imagen
    image = np.array(image, dtype=np.uint8)  # Array de 8 bits (0, 255)

    # Redimensiona la imagen una sola vez
    image1 = cv2.resize(image, dsize=(224, 224))

    # Preprocesamiento de datos
    image_arr_224 = img_to_array(image1) / 255.0  # Convierte en un array y normaliza la salida
    h, w, d = image.shape
    test_arr = image_arr_224.reshape(1, 224, 224, 3)

    # Realiza predicciones
    coords = model.predict(test_arr)

    # Denormaliza los valores
    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    return coords

def OCR(image):
    """
    Realiza el reconocimiento óptico de caracteres (OCR) en una región de interés de la imagen.

    Parameters:
    image (numpy.ndarray): La imagen de entrada en formato NumPy.

    Returns:
    str: El texto reconocido después de aplicar filtros de procesamiento de imagen.
    """
    img = np.array(image)
    cods = object_detection(img.copy())
    xmin, xmax, ymin, ymax = cods[0]
    roi = img[ymin:ymax, xmin:xmax]

    # Aplica filtros de procesamiento de imagen
    roi = cv2.fastNlMeansDenoisingColored(roi, None, 10, 10, 7, 21)
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    roi = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, rectKern)

    # Realiza OCR en la región de interés
    text = pt.image_to_string(roi)
    text = "".join([i for i in text if i in KEYS])

    return text

# Lista todos los archivos en la carpeta
folder_path = "./placa"
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Asegúrate de incluir los formatos de imagen que necesitas
        # Carga la imagen
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Aplica tus funciones de detección de objetos y OCR
        detected_objects = object_detection(image)
        text = OCR(image)

        # Muestra los resultados
        print(f"Imagen: {filename}")
        print("Objetos detectados:", detected_objects)
        print("Texto OCR:", text)
"""
Este código carga un modelo de detección de objetos previamente entrenado y luego aplica tanto la detección de objetos 
como el reconocimiento óptico de caracteres (OCR) a las imágenes en una carpeta especificada. Los comentarios en el código
explican las funciones y los pasos clave.
"""