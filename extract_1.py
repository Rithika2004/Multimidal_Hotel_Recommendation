
# Part 1: Data extraction pipeline (keep from your previous code)
import pytesseract
from PIL import Image
import requests
from bs4 import BeautifulSoup

# Extract text from image
image_path = "image_1.jpg"
image = Image.open(image_path)
ocr_text = pytesseract.image_to_string(image)

# Identify likely website from keywords in text
search_keywords = []
if "Qmin" in ocr_text:
    search_keywords.append("Qmin")
if "Ginger" in ocr_text or "Ginger Hotels" in ocr_text:
    search_keywords.append("Ginger Hotels")

websites = {
    "Qmin": "https://qmin.co.in/",
    "Ginger Hotels": "https://gingerhotels.com/",
}
selected_website = None
for keyword in search_keywords:
    if keyword in websites:
        selected_website = websites[keyword]
        break

if selected_website:
    try:
        response = requests.get(selected_website)
        soup = BeautifulSoup(response.text, "html.parser")
        site_text = soup.get_text(separator=' ', strip=True)
        website_summary = site_text[:1000]
    except Exception as e:
        website_summary = f"Failed to fetch website data: {e}"
else:
    website_summary = "No relevant hotel website found in extracted text!"

hotel_data = {
    "hotel_name": "Detected from image (e.g., Qmin at Ginger Hotels)",
    "image_text": ocr_text,
    "website_url": selected_website if selected_website else "",
    "website_summary": website_summary,
    "tagline": "Good Food, Good Vibes",
    "average_cost_for_two": "₹1000",
}

# --- Part 2: Multimodal embedding, clustering, and recommendation ---
import clip
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re # Added for text cleaning

# Placeholder values for demonstration (You need to run Part 1 code to get these)
# Assuming hotel_data is available in the environment from Part 1
# For a runnable script, we would need to ensure these are defined:
try:
    # Attempt to use variables defined globally by the user's Part 1 script
    hotel_data
    image_path
except NameError:
    # Define minimal placeholder data if Part 1 code is not run alongside
    class MockClip:
        @staticmethod
        def tokenize(texts):
            # Mocking the token output structure for safety
            return torch.tensor([[1] * min(77, len(texts[0])//10)]).unsqueeze(0)

    clip = MockClip()
    image_path = "placeholder_image.jpg"
    hotel_data = {
        "image_text": "Sample image text.",
        "website_summary": "Sample website summary that might be very long.",
        "hotel_name": "Sample Hotel",
    }


device = "cuda" if torch.cuda.is_available() else "cpu"
# Only load CLIP if the environment allows or if we are not in a mock setup
if 'MockClip' not in locals():
    model, preprocess = clip.load("ViT-B/32", device=device)
else:
    print("Warning: Using Mock CLIP setup as environment is incomplete.")

# --- Text embedding, robust token limit solution (Token-based Truncation) ---
def truncate_text_to_77_tokens(text, clip_module):
    # 1. Robustly sanitize the text
    text = text.replace('\n', ' ').replace('\t', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Failed to fetch website data:', '', text, flags=re.IGNORECASE)
    # Aggressive cleaning of potential long error strings and unnecessary web artifacts
    text = re.sub(r'403 Forbidden.*$', '', text, flags=re.IGNORECASE).strip() 
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Remove non-ASCII characters that can mess up tokenizers

    if not text:
        return ""

    # CRITICAL FIX: Aggressive character pre-truncation to prevent C++ buffer error.
    # The limit is arbitrary but guarantees the string isn't excessively long for the tokenizer's internal buffer.
    # We choose a small number (512) to be absolutely safe, as 77 tokens rarely correspond to more than 350-400 characters.
    MAX_CHAR_PRE_TRUNCATION = 512
    if len(text) > MAX_CHAR_PRE_TRUNCATION:
        text = text[:MAX_CHAR_PRE_TRUNCATION] 
    
    # --- Rerunning the Binary Search with a safer loop condition ---
    left, right = 0, len(text)
    best_candidate = ""
    
    while left <= right:
        mid = (left + right) // 2
        candidate = text[:mid]
        
        # Check token count for the current prefix
        try:
            tokens = clip_module.tokenize([candidate])
            token_count = tokens.shape[1]
        except RuntimeError:
            # If tokenizing the candidate string fails (extremely rare after pre-truncation), shorten it.
            right = mid - 1
            continue
            
        if token_count <= 77:
            # Current candidate is valid, save it and try a longer string
            best_candidate = candidate
            left = mid + 1
        else:
            # Too long, shorten the string
            right = mid - 1
            
    # Final cleanup of partial words
    if best_candidate and best_candidate[-1].isalnum() and len(text) > len(best_candidate) and text[len(best_candidate)].isalnum():
        best_candidate = ' '.join(best_candidate.split(' ')[:-1])
    
    return best_candidate.strip()


# --- Execution continues only if CLIP model is loaded ---
if 'model' in locals():
    import streamlit as st 
    import pandas as pd 
    
    # Image embedding
    # Ensure image_path points to a valid image before running
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}. Cannot generate image features.")
        # If image features fail, use a zero vector placeholder (or handle error gracefully)
        image_features = torch.zeros((1, 512), device=device)

    joined_text = f"{hotel_data['image_text']} {hotel_data['website_summary']}"
    
    # Note: If the RuntimeError is still raised here, it means the input string 
    # is fundamentally incompatible with the CLIP tokenizer's maximum buffer size.
    # The aggressive text sanitization should have fixed this.
    short_text = truncate_text_to_77_tokens(joined_text, clip)
    
    # Final attempt to tokenize the verified short_text
    try:
        text_tokens = clip.tokenize([short_text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
    except RuntimeError as e:
        print(f"FATAL ERROR: Final tokenization of short_text failed. Error: {e}")
        print(f"Problematic Text Snippet: {short_text[:200]}...")
        # Fallback to zero vector to prevent crash
        text_features = torch.zeros((1, 512), device=device)


    # --- Joint multimodal embedding ---
    joint_embedding = (image_features.cpu().numpy() + text_features.cpu().numpy()) / 2  # shape (1, 512)

    # -- Create dataset
    X = joint_embedding  # shape (N, 512)

    # Simulate dataset with 10 identical samples (for demo)
    X_repeat = np.repeat(X, repeats=10, axis=0)  # Fake 10 entries

    # --- Clustering ---
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto') # Use 'auto' initialization
    labels = kmeans.fit_predict(X_repeat)
    segments = {
        0: "Luxury Experience",
        1: "Comfort Ambience",
        2: "Quick Eats"
    }
    segment_assignments = [segments[label] for label in labels]

    print("Hotel segments assigned:", segment_assignments)

    # --- Recommendation function ---
    def recommend_hotel(user_query, centroids, hotel_names, hotel_texts):
        query_text = truncate_text_to_77_tokens(user_query, clip)
        with torch.no_grad():
            # Using the same final tokenization safety block
            try:
                query_features = model.encode_text(clip.tokenize([query_text]).to(device)).cpu().numpy()
            except RuntimeError:
                query_features = np.zeros((1, 512))
                
        sims = cosine_similarity(query_features, centroids)
        top_index = np.argmax(sims)
        # We return the segment name associated with the centroid
        return segments[top_index], hotel_texts[top_index]

    # Example usage:
    hotel_names = [hotel_data["hotel_name"]] * 10
    hotel_texts = [short_text] * 10 
    recommended_segment, recommended_text = recommend_hotel(
        "Comfort food in lively ambience",
        kmeans.cluster_centers_,
        hotel_names,
        hotel_texts
    )
    print(f"Recommended segment: {recommended_segment}")
    print(f"Text used for embedding: {short_text[:100]}...")
    
# --- The Streamlit UI block has been moved outside the 'if model in locals()' check
# but still requires the variables defined within that block. For a standard environment,
# these variables (recommended_segment, recommended_text, segment_assignments, etc.)
# would be available globally after execution.

# 1. Setup and Aesthetics
    
        # --- HELPER FUNCTION DEFINED HERE TO AVOID NAMERROR ---
    def clean_display_text(raw_text):
        if not raw_text:
            return "N/A or Empty Content"
        # Aggressive removal of common OCR junk and server errors
        cleaned = re.sub(r'(OOD M\d+)', '', raw_text, flags=re.IGNORECASE)
        cleaned = re.sub(r'403 Forbidden.*', 'Website Access Restricted/Failed.', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'Tanes as app', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'isit Qmin', 'Qmin', cleaned, flags=re.IGNORECASE) # Fix common OCR mistake
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned if len(cleaned) > 5 else "Content not informative." # Filter out near-empty strings


    # --- Streamlit UI ---
    
    # 1. Setup and Aesthetics
    st.set_page_config(page_title="Multimodal Hotel Recommendation Dashboard", layout="wide")
    
    # 2. Main Title and Navigation
    st.title("🏨 Multimodal Hotel Segment & Recommendation Dashboard")
    st.markdown("Powered by CLIP Embedding and K-Means Clustering.")
    
    # Initialize session state for navigation if not present
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'
    
    # --- SIDEBAR: Navigation and Metadata ---
    st.sidebar.header("Navigation")
    
    # 1. Move Home and Contact Us buttons to the sidebar
    if st.sidebar.button("🏠 Home"):
        st.session_state.current_page = 'Home'
    if st.sidebar.button("📧 Contact Us"):
        st.session_state.current_page = 'Contact'
    
    # --- Contact Page ---
    if st.session_state.current_page == 'Contact':
        st.header("Contact the Development Team")
        st.info("For queries regarding the Multimodal Recommendation System, please contact:")
        st.markdown("- **Team Lead:** Rithika Venkatesan")
        st.markdown("- **Email:** rithika.venkatesan@kgpian.iitkgp.ac.in")
        st.markdown("- **Institute:** IIT Kharagpur")

    # --- Home Page (Main Dashboard Content) ---
    if st.session_state.current_page == 'Home':
        
        # 3. Hotel Selection Dropdown
        available_hotels = [hotel_data['hotel_name']] # In a real app, this would be a list of all hotel names
        selected_hotel = st.selectbox(
            "🔍 Select a Hotel to Analyze:",
            options=['--- Select Hotel ---'] + available_hotels,
            index=0 # Initially show prompt
        )

        # 1. Stop Pre-display of Sidebar Data: Moving the metadata display inside this block.
        if selected_hotel != '--- Select Hotel ---':
                
            # 4. Heading Change based on Selection
            st.header(f"Hotel Overview") # Renamed heading

            # --- MAIN CONTENT LAYOUT ---
            
            # Row 1: Extracted Features and Segmentation
            col1, col2 = st.columns([2, 1]) 

            with col1:
                
                # Display Tagline and Cost (Point 5)
                st.markdown(f"**Tagline:** **{hotel_data.get('tagline', 'N/A')}**")
                st.markdown(f"**Average Cost for Two:** {hotel_data.get('average_cost_for_two', 'N/A')}")
                
                st.markdown(f"**🔍 Extracted Text from Hotell Ad (OCR)**")
                # Display CLEANED text
                cleaned_image_text = clean_display_text(hotel_data['image_text'])
                st.markdown(cleaned_image_text.replace('\n', ' '))
                
                # 2. Website Context Snippet dropdown entirely removed.
                st.markdown(f"**Click here to visit the website:** ")
                if hotel_data.get('website_url'):
                    st.markdown(f"[**Visit Hotel Website**]({hotel_data['website_url']})") # Display visit website link directly here
            
            with col2:
                # 6. Rename Clustering result to "Categories Matched"
                st.subheader("Categories Matched")
                # Display segments clearly using st.metric for emphasis
                st.metric("Assigned Category", f"{segment_assignments[0]}", delta_color="off")
                
                st.caption("All Matching Categories:")
                # Display all possible segments
                segment_data = pd.DataFrame(segments.items(), columns=["ID", "Category Type"]) # Renamed column
                st.dataframe(segment_data.set_index('ID'), use_container_width=True, hide_index=True)


        st.markdown("---")
        st.caption("Multimodal Model Status: ViT-B/32 on CUDA/CPU.")
else:
    # Ensure Streamlit is defined for the UI if the main code failed to run
    import streamlit as st
    st.error("Code halted due to inability to load CLIP or missing dependencies. Cannot display results.")