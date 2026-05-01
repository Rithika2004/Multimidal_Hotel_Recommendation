# search_engine.py - Loads vectors, computes cosine sim for user query
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

def load_vectors():
    """Load precomputed CLIP vectors from vector_store.pkl."""
    try:
        with open('vector_store.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['vectors'], data['metadata'], data['dataset']
    except FileNotFoundError:
        print("Run preprocessor first to create vector_store.pkl")
        return np.array([]), [], []

def query_to_embedding(query_text, truncate=True):
    """Mock CLIP text embedding (replace with real CLIP in extract_1.py)."""
    # In full app, use: model.encode_text(clip.tokenize([preprocess_text(query_text)]))
    # Mock 512-dim vector for demo (deterministic based on query)
    hash_val = hash(query_text) % 1000 / 1000.0
    return np.random.RandomState(int(hash_val * 100)).rand(512).astype(np.float32).reshape(1, -1)

def recommend(query, top_k=5):
    """Find top hotel ads by cosine similarity."""
    vectors, metadata, dataset = load_vectors()
    if len(vectors) == 0:
        return []
    query_emb = query_to_embedding(query)
    sims = cosine_similarity(query_emb, vectors)[0]
    top_indices = np.argsort(sims)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'hotel': dataset[idx]['name'],
            'text': metadata[idx],
            'similarity': sims[idx],
            'image_ref': dataset[idx].get('image_ref', '')
        })
    return results

# Example
print(recommend("luxury pool suite cheap")[0])
