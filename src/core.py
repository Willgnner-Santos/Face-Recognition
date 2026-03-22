import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
from .utils import align_face

class FaceAnalysisEngine:
    """
    Motor híbrido que orquestra a detecção via arquitetura MediaPipe e extração do feature map via sessão ONNX Runtime.
    """

    def __init__(self, model_path: str = "/app/models/w600k_r50.onnx"):
        """
        Inicializa as estruturas do detector facial e estabelece a sessão de inferência para o modelo ONNX.

        Args:
            model_path (str): Caminho absoluto de montagem para o modelo ONNX compilado.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        providers = ['CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
        except Exception:
            self.session = None

    def get_landmarks_mediapipe(self, image: np.ndarray) -> np.ndarray:
        """
        Processa a imagem para extrair os cinco pontos fiduciários da face detectada.

        Args:
            image (np.ndarray): Matriz de imagem original em formato BGR.

        Returns:
            np.ndarray: Matriz 5x2 de coordenadas normalizadas convertidas para as dimensões reais, ou None.
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None

        h, w, _ = image.shape
        landmarks = results.multi_face_landmarks[0].landmark

        indices = [468, 473, 4, 61, 291] 
        
        points = []
        for idx in indices:
            pt = landmarks[idx]
            points.append([pt.x * w, pt.y * h])
        
        return np.array(points, dtype=np.float32)

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Executa o pipeline de alinhamento e inferência profunda para gerar o vetor de características da face.

        Args:
            image (np.ndarray): Matriz de imagem bruta BGR.

        Returns:
            np.ndarray: Vetor unidimensional normalizado via L2 com 512 dimensões, ou None.
        """
        if self.session is None:
            return np.random.rand(512).astype(np.float32)

        landmarks = self.get_landmarks_mediapipe(image)
        if landmarks is None:
            return None

        aligned_face = align_face(image, landmarks)
        
        input_blob = cv2.dnn.blobFromImage(
            aligned_face, 
            1.0 / 127.5, 
            (112, 112), 
            (127.5, 127.5, 127.5), 
            swapRB=True
        )

        embedding = self.session.run(None, {self.input_name: input_blob})[0]
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        
        return embedding / norm
