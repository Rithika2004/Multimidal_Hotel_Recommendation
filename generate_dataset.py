# generate_dataset.py - Offline embedding + save (adapt to your 50 images)
import pickle
import numpy as np
# pip install torch torchvision clip-by-openai pillow scikit-learn
import torch
import clip
from PIL import Image
from preprocessor import preprocess_text  # From above

# Your 50 hotel ad images (download from [image:11]-[image:23], name image_1.jpg etc.)
image_paths = ['image_1.jpg', 'image_2.jpg'] * 25  # Placeholder; replace with your files
texts = ['luxury hotel promo pool suite $150'] * 50  # OCR texts

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

vectors = []
metadata = []
dataset = []

for i, path in enumerate(image_paths[:50]):
    # Image embedding
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = model.encode_image(image).cpu().numpy()
    
    # Text embedding (preprocessed OCR)
    clean_text = preprocess_text(texts[i])
    text_input = clip.tokenize([clean_text[:512]]).to(device)  # Safe truncate
    with torch.no_grad():
        txt_emb = model.encode_text(text_input).cpu().numpy()
    
    # Joint multimodal
    joint = (img_emb + txt_emb) / 2
    vectors.append(joint[0])
    
    metadata.append(clean_text)
    dataset.append({'id': f'hotel_{i+1}', 'name': f'Hotel {i+1}', 'text': texts[i], 'image_ref': f'image:{11+i%13+1}'})

vectors = np.array(vectors)
with open('vector_store.pkl', 'wb') as f:
    pickle.dump({'vectors': vectors, 'metadata': np.array(metadata), 'dataset': dataset}, f)

print(f'Saved 50 hotel ads: {vectors.shape}')
