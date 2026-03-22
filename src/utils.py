import base64
import cv2
import numpy as np
from typing import List

def decode_image(base64_string: str) -> np.ndarray:
    """
    Decodifica uma string base64 para um array numpy em formato BGR.
    Aplica redimensionamento se a dimensão máxima ultrapassar 640 pixels para otimização de processamento.

    Args:
        base64_string (str): String da imagem codificada.

    Returns:
        np.ndarray: Imagem decodificada e possivelmente redimensionada.
    """
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    height, width = image.shape[:2]
    max_dimension = 640

    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image

def generate_brightness_variants(image: np.ndarray) -> List[np.ndarray]:
    """
    Gera variantes de luminosidade da matriz de imagem original.

    Args:
        image (np.ndarray): Matriz de imagem original BGR.

    Returns:
        List[np.ndarray]: Lista contendo a imagem original, variante com adição de brilho e variante com subtração de brilho.
    """
    bright_matrix = np.ones(image.shape, dtype="uint8") * 40
    bright_img = cv2.add(image, bright_matrix)
    dark_img = cv2.subtract(image, bright_matrix)

    return [image, bright_img, dark_img]

def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Aplica transformação afim parcial bidimensional para alinhar a face extraída com base em cinco pontos fiduciários.

    Args:
        image (np.ndarray): Matriz de imagem original BGR.
        landmarks (np.ndarray): Matriz 5x2 contendo as coordenadas faciais em ponto flutuante.

    Returns:
        np.ndarray: Face recortada e geometricamente alinhada na dimensão de 112x112 pixels.
    """
    dst_pts = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.6963],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.3655]
    ], dtype=np.float32)

    tform = cv2.estimateAffinePartial2D(landmarks, dst_pts)[0]
    output = cv2.warpAffine(image, tform, (112, 112))
    return output
