import os
import shutil
import streamlit as st
import pandas as pd
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Menghapus folder yang tidak valid (punkt_tab) jika ada
try:
    punkt_tab_dir = '/home/appuser/nltk_data/tokenizers/punkt_tab'
    if os.path.exists(punkt_tab_dir):
        shutil.rmtree(punkt_tab_dir)  # Menghapus folder dan isinya
        st.info(f"Folder {punkt_tab_dir} berhasil dihapus.")
    else:
        st.info(f"Folder {punkt_tab_dir} tidak ditemukan.")
except Exception as e:
    st.error(f"Error during folder deletion: {e}")

# Pastikan resource NLTK tersedia
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")

# Load stopwords and stemmer
try:
    stop_words = set(stopwords.words('indonesian'))
except LookupError:
    stop_words = set()
stemmer = PorterStemmer()

# Load models and resources
try:
    models = {
    "liputan6_gempa": joblib.load('liputan6_gempa.pkl'),
    "liputan6_banjir": joblib.load('liputan6_banjir.pkl'),
    "liputan6_tsunami": joblib.load('liputan6_tsunami.pkl'),
    "detik_gempa": joblib.load('detik_gempa.pkl'),
    "detik_banjir": joblib.load('detik_banjir.pkl'),
    "detik_tsunami": joblib.load('detik_tsunami.pkl'),
    "tribunnews_gempa": joblib.load('tribunnews_gempa.pkl'),
    "tribunnews_banjir": joblib.load('tribunnews_banjir.pkl'),
    "tribunnews_tsunami": joblib.load('tribunnews_tsunami.pkl'),
    "okezonenews_gempa": joblib.load('okezonenews_gempa.pkl'),
    "okezonenews_banjir": joblib.load('okezonenews_banjir.pkl'),
    "okezonenews_tsunami": joblib.load('okezonenews_tsunami.pkl'),
    "BMKG_gempa": joblib.load('BMKG_gempa.pkl')
    }

    vectorizers = {
        "liputan6_gempa": joblib.load('liputan6_gempa_vectorizer.pkl'),
        "liputan6_banjir": joblib.load('liputan6_banjir_vectorizer.pkl'),
        "liputan6_tsunami": joblib.load('liputan6_tsunami_vectorizer.pkl'),
        "detik_gempa": joblib.load('detik_gempa_vectorizer.pkl'),
        "detik_banjir": joblib.load('detik_banjir_vectorizer.pkl'),
        "detik_tsunami": joblib.load('detik_tsunami_vectorizer.pkl'),
        "tribunnews_gempa": joblib.load('tribunnews_gempa_vectorizer.pkl'),
        "tribunnews_banjir": joblib.load('tribunnews_banjir_vectorizer.pkl'),
        "tribunnews_tsunami": joblib.load('tribunnews_tsunami_vectorizer.pkl'),
        "okezonenews_gempa": joblib.load('okezonenews_gempa_vectorizer.pkl'),
        "okezonenews_banjir": joblib.load('okezonenews_banjir_vectorizer.pkl'),
        "okezonenews_tsunami": joblib.load('okezonenews_tsunami_vectorizer.pkl'),
        "BMKG_gempa": joblib.load('BMKG_gempa_vectorizer.pkl')
    }
except Exception as e:
    st.error(f"Error loading models or vectorizers: {e}")

# Load train data
train_data_files = [
    'liputan6_gempa_tokenized&stemmed.csv', 'liputan6_banjir_tokenized&stemmed.csv', 'liputan6_tsunami_tokenized&stemmed.csv',
    'detik_gempa_tokenized&stemmed.csv', 'detik_banjir_tokenized&stemmed.csv', 'detik_tsunami_tokenized&stemmed.csv',
    'tribunnews_gempa_tokenized&stemmed.csv', 'tribunnews_banjir_tokenized&stemmed.csv', 'tribunnews_tsunami_tokenized&stemmed.csv',
    'okezonenews_gempa_tokenized&stemmed.csv', 'okezonenews_banjir_tokenized&stemmed.csv', 'okezonenews_tsunami_tokenized&stemmed.csv',
    'BMKG_gempa_tokenized&stemmed.csv'
]

train_texts = []
for file in train_data_files:
    try:
        data = pd.read_csv(file)
        train_texts.extend(data['full_text'].dropna().tolist())
    except Exception as e:
        st.warning(f"Error loading {file}: {e}")

def preprocess_text(text):
    """Preprocessing untuk teks input."""
    try:
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in stop_words]
        words = [stemmer.stem(word) for word in words]
        return ' '.join(words)
    except Exception as e:
        st.error(f"Error during text preprocessing: {e}")
        return ""

def predict_text(input_text, model, vectorizer, train_texts):
    """Prediksi objektivitas teks input."
