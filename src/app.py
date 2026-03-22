import os
import time
import uuid
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from .core import FaceAnalysisEngine
from .utils import decode_image, generate_brightness_variants

app = Flask(__name__)

qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant = QdrantClient(host=qdrant_host, port=6333)
COLLECTION_NAME = "faces_db"
engine = FaceAnalysisEngine(model_path="/app/models/w600k_r50.onnx")

def init_db():
    """
    Valida e assegura a construção da coleção de armazenamento vetorial no banco de dados Qdrant.
    Implementa tolerância a falhas para inicialização assíncrona do serviço vetorial.
    """
    max_retries = 10
    
    for _ in range(max_retries):
        try:
            collections_response = qdrant.get_collections()
            exists = any(col.name == COLLECTION_NAME for col in collections_response.collections)
            
            if not exists:
                qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
                )
            return
        except Exception:
            time.sleep(3)
            
    raise ConnectionError("Falha crítica: Qdrant inacessível após múltiplas tentativas.")

with app.app_context():
    init_db()

@app.route('/register', methods=['POST'])
def register_face():
    """
    Rota de ingestão que decodifica o payload base64, gera matrizes de variabilidade e persiste os vetores.

    Returns:
        Response: Objeto JSON contendo a mensagem de estado e a quantidade absoluta de vetores injetados na coleção.
    """
    data = request.get_json()
    name = data.get("name")
    img_b64 = data.get("image")

    if not name or not img_b64:
        return jsonify({"error": "Parâmetros nome e imagem ausentes"}), 400

    original_img = decode_image(img_b64)
    variants = generate_brightness_variants(original_img)
    
    points = []
    
    for idx, img in enumerate(variants):
        embedding = engine.get_embedding(img)
        
        if embedding is not None:
            payload = {"name": name}
            if idx == 0:
                payload["original_image_b64"] = img_b64

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=payload
            ))

    if not points:
        return jsonify({"error": "Falha na detecção facial"}), 400

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    return jsonify({
        "message": "Registro efetuado", 
        "vectors_stored": len(points)
    }), 201

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """
    Rota de inferência estrita que mapeia a métrica de distância cosseno entre a face de entrada e o banco de dados.

    Returns:
        Response: Objeto JSON contendo o nome recuperado, ou a designação Unknown baseada no threshold limite, e o score matemático.
    """
    data = request.get_json()
    img_b64 = data.get("image")
    
    THRESHOLD = 0.5 

    if not img_b64:
        return jsonify({"error": "Payload inválido"}), 400

    img = decode_image(img_b64)
    embedding = engine.get_embedding(img)

    if embedding is None:
        return jsonify({"result": "No Face Detected"}), 200

    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding.tolist(),
        limit=1,
        score_threshold=THRESHOLD
    )

    if search_result:
        best_match = search_result[0]
        return jsonify({
            "result": best_match.payload["name"],
            "score": float(best_match.score)
        }), 200
    
    return jsonify({"result": "Unknown", "score": 0.0}), 200

@app.route('/reset', methods=['DELETE'])
def reset_collection():
    """
    Remove a coleção vetorial atual do Qdrant e inicializa uma nova estrutura vazia.

    Returns:
        Response: Objeto JSON contendo a confirmação do expurgo e recriação da base.
    """
    qdrant.delete_collection(collection_name=COLLECTION_NAME)
    init_db()
    return jsonify({"message": "Banco de dados limpo com sucesso"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)