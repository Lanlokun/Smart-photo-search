from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import faiss
import torch
from typing import List, Tuple, Dict
from app.models import Photo

model = SentenceTransformer('clip-ViT-B-32')

def get_image_embedding(img_path: str) -> np.ndarray:
    """Extracts and normalizes embedding from image file."""
    image = Image.open(img_path).convert("RGB")
    return model.encode(image, convert_to_numpy=True, normalize_embeddings=True).astype('float32')

def preload_image_embeddings(photo_objs: List[Photo]) -> Dict[Photo, np.ndarray]:
    """Loads and normalizes stored embeddings from DB Photo objects."""
    embeddings = {}
    for photo in photo_objs:
        try:
            emb_array = photo.get_embedding()
            if emb_array is not None:
                emb_np = np.asarray(emb_array, dtype='float32')
                norm = np.linalg.norm(emb_np)
                if norm > 0:
                    emb_np /= norm
                embeddings[photo] = emb_np
        except Exception as e:
            print(f"[Embedding Load Error] {photo.filename}: {e}")
    return embeddings

def build_faiss_index(embeddings: Dict[Photo, np.ndarray]) -> Tuple[faiss.IndexFlatIP, List[Photo]]:
    """Builds a FAISS index from photo embeddings."""
    if not embeddings:
        return None, []

    photo_list = list(embeddings.keys())
    matrix = np.stack([embeddings[photo] for photo in photo_list])

    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    return index, photo_list

def fast_search(
    query: str,
    index: faiss.IndexFlatIP,
    photo_list: List[Photo],
    top_k: int = 12
) -> List[Tuple[Photo, float]]:
    """Searches the FAISS index for photos semantically similar to query."""
    if not index or not photo_list:
        return []

    query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype('float32').reshape(1, -1)

    distances, indices = index.search(query_embedding, top_k)
    results = []

    for rank, idx in enumerate(indices[0]):
        if 0 <= idx < len(photo_list):
            results.append((photo_list[idx], float(distances[0][rank])))

    return results
