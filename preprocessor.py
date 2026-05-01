# preprocessor.py - Cleans OCR text with lemmatization (free NLTK)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(raw_text):
    """Clean OCR junk, lemmatize, remove stopwords."""
    if not raw_text:
        return ""
    # Remove junk like 'OOD M54', errors
    cleaned = re.sub(r'(OOD M\d+)|403 Forbidden|Failed to fetch|Tanes as app', '', raw_text, flags=re.IGNORECASE)
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)  # Non-ASCII
    cleaned = re.sub(r'\s+', ' ', cleaned.strip().lower())
    words = [lemmatizer.lemmatize(w) for w in cleaned.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# Example usage
sample_ocr = "OOD M54 Qmin Ginger Failed to fetch 403 good food vibes ₹1000"
print("Cleaned:", preprocess_text(sample_ocr))
