import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
import os
import time
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from scipy.sparse import save_npz, load_npz
import pickle

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Category Mapping
category_mapping = {
    'Credit reporting, credit repair services, or other personal consumer reports': 0,
    'Debt collection': 1,
    'Consumer Loan': 2,
    'Mortgage': 3
}

# File paths
RAW_FILE = "complaints.csv"
CLEANED_FILE = "complaints_cleaned2.csv"
TFIDF_MATRIX_FILE = "tfidf_features.npz"
TFIDF_VECTORIZER_FILE = "tfidf_vectorizer.pkl"
OUTPUT_DIR = "preprocessed_data"


def log_progress(message: str):
    current_time = time.strftime('%H:%M:%S', time.localtime())
    print(f"[{current_time}] {message}")
    sys.stdout.flush()

def safe_create_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        log_progress(f"Created directory: {directory}")


def load_and_preprocess_data():
    Target_Column = "Product"
    Text_Column = "Consumer complaint narrative"

    log_progress("Checking file existence")
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"File {RAW_FILE} not found.")

    log_progress("Loading CSV file")
    df = pd.read_csv(RAW_FILE, usecols=[Target_Column, Text_Column], low_memory=False)

    log_progress("Dropping NaN values")
    df.dropna(subset=[Text_Column, Target_Column], inplace=True)

    log_progress("Filtering only selected categories")
    df = df[df[Target_Column].isin(category_mapping.keys())]
    df['label'] = df[Target_Column].map(category_mapping)

    log_progress("Initial class distribution:")
    print(df['label'].value_counts())

    return df

def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def preprocess_text_column(df):
    log_progress("Preprocessing text column")
    tqdm.pandas(desc="Cleaning text")
    df['cleaned_text'] = df['Consumer complaint narrative'].progress_apply(text_preprocessing)

    log_progress("Handling empty strings after preprocessing")
    df = df[df['cleaned_text'].str.strip() != ""]

    nan_count = df['cleaned_text'].isnull().sum()
    if nan_count > 0:
        log_progress(f"Found {nan_count} NaN values in cleaned_text. Filling with empty strings.")
        df['cleaned_text'] = df['cleaned_text'].fillna("")

    return df

#Generate TF-IDF features or load precomputed ones if they exist.
def generate_or_load_tfidf(df):
    safe_create_dir(OUTPUT_DIR)

    tfidf_matrix_path = os.path.join(OUTPUT_DIR, TFIDF_MATRIX_FILE)
    vectorizer_path = os.path.join(OUTPUT_DIR, TFIDF_VECTORIZER_FILE)


    if os.path.exists(tfidf_matrix_path) and os.path.exists(vectorizer_path):
        try:
            log_progress("Precomputed TF-IDF found. Loading from disk...")
            X_tfidf = load_npz(tfidf_matrix_path)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            log_progress(f"TF-IDF loaded successfully: shape {X_tfidf.shape}")
            return X_tfidf, vectorizer
        except Exception as e:
            log_progress(f"Failed to load precomputed TF-IDF: {e}")
            log_progress("Regenerating TF-IDF from scratch...")

    log_progress("Generating TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.9
    )

    start = time.time()
    X_tfidf = vectorizer.fit_transform(df['cleaned_text'])
    end = time.time()

    log_progress(f"TF-IDF completed in {end - start:.2f} seconds.")
    log_progress(f"TF-IDF matrix shape: {X_tfidf.shape}")

    try:
        save_npz(tfidf_matrix_path, X_tfidf)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        log_progress(f"TF-IDF matrix saved to {tfidf_matrix_path}")
        log_progress(f"Vectorizer saved to {vectorizer_path}")
    except Exception as e:
        log_progress(f"Failed to save TF-IDF data: {e}")

    return X_tfidf, vectorizer


def main():
    start_time = time.time()
    try:
        df = load_and_preprocess_data()
        df = preprocess_text_column(df)

        log_progress("Class distribution after cleaning:")
        print(df['label'].value_counts())

        X_tfidf, vectorizer = generate_or_load_tfidf(df)

        # Save cleaned CSV
        safe_create_dir(OUTPUT_DIR)
        cleaned_path = os.path.join(OUTPUT_DIR, CLEANED_FILE)
        df.to_csv(cleaned_path, index=False)
        log_progress(f"Cleaned data saved to {cleaned_path}")

        total_time = time.time() - start_time
        log_progress(f"Preprocessing pipeline completed in {total_time:.2f} seconds.")

    except Exception as e:
        log_progress(f"Pipeline failed due to error: {e}")

if __name__ == "__main__":
    main()

