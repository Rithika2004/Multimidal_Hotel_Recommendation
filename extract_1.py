import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Mock dataset (50 hotels)
@st.cache_data
def load_dataset():
    hotel_dataset = [{'id': f'hotel_{i}', 'name': f'Hotel {i}', 'text': f'Luxury hotel ad poster {i} pool suite meals $150 book now promo'} for i in range(1,51)]
    # Mock 512-dim embeddings (hash-based)
    np.random.seed(42)
    vectors = np.random.rand(50, 512).astype(np.float32)
    metadata = np.array([h['text'] for h in hotel_dataset])
    return vectors, metadata, hotel_dataset

def text_embedding(text):
    """Mock CLIP text embedding (numpy hash)."""
    seed = int(hash(text) % 2**32)
    np.random.seed(seed)
    return np.random.rand(512).astype(np.float32).reshape(1, -1)

st.set_page_config(page_title="Hotel Recommendation", layout="wide")
st.title("🏨 Multimodal Hotel Recommendation Dashboard")
st.markdown("Text-based demo (CLIP mock for Python 3.14 compatibility)")

vectors, metadata, hotels = load_dataset()

tab1, tab2 = st.tabs(["Query Hotels", "All Hotels"])

with tab1:
    query = st.text_input("Your preference (e.g., 'luxury pool cheap')")
    if query:
        q_emb = text_embedding(query)
        sims = cosine_similarity(q_emb, vectors)[0]
        top_k = np.argsort(sims)[-5:][::-1]
        st.subheader("Top 5 Recommendations")
        for i, idx in enumerate(top_k):
            st.metric(f"#{i+1} {hotels[idx]['name']}", f"{sims[idx]:.2f}", delta="Match")

with tab2:
    st.dataframe(pd.DataFrame({'Hotel': [h['name'] for h in hotels], 'Ad Text': metadata}))

st.caption("Deployed on Streamlit Cloud | 50 hotel ads ready.")
