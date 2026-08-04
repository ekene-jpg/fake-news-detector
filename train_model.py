"""
Model training script — implements Chapter 3, sections 3.3.3(a)-(c), 3.7, 3.8.1, 3.14, 3.15.

Pipeline: clean_text -> TF-IDF (max_features=5000) -> Logistic Regression
-> 80/20 train/test split -> accuracy/precision/recall/F1.

Usage:
    python train_model.py
Reads data/news_dataset.csv (columns: title, text, subject, date, label)
and writes model/model.pkl + model/vectorizer.pkl.
"""
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- NLTK stopwords (section 3.3.3a) ---
# In your real environment (with internet access), this downloads once and caches.
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        STOPWORDS = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        STOPWORDS = set(stopwords.words('english'))
except ImportError:
    # This sandbox has no internet, so nltk can't be installed here.
    # Falls back to scikit-learn's built-in English stopword list so the
    # demo can still run end-to-end. On your machine, `pip install nltk`
    # (see requirements.txt) will use the real nltk corpus instead.
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOPWORDS = set(ENGLISH_STOP_WORDS)
    print("[note] nltk not available in this environment — using sklearn's "
          "built-in stopword list instead. Install nltk on your machine for "
          "the exact pipeline described in Chapter 3.")


def clean_text(text):
    """Section 3.3.3(a): lowercase, strip non-letters, remove stopwords."""
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return ' '.join(words)


def main():
    # --- 3.4 Data Collection ---
    df = pd.read_csv("data/news_dataset.csv")
    print(f"Loaded {len(df)} rows: {(df.label==0).sum()} fake, {(df.label==1).sum()} real")

    # Combine title + text, as most Kaggle Fake News pipelines do
    corpus_raw = (df['title'].fillna('') + '. ' + df['text'].fillna(''))

    # --- 3.6 Data Preprocessing ---
    print("Cleaning text...")
    corpus = corpus_raw.apply(clean_text)
    y = df['label'].values

    # --- 3.7 Feature Extraction (TF-IDF) ---
    print("Vectorizing (TF-IDF, max_features=5000)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(corpus).toarray()

    # --- 3.14 Model Training (80/20 split) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # --- 3.15 Model Evaluation ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Evaluation (held-out 20% test set) ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"Confusion matrix:\n{cm}")

    # --- Save artifacts ---
    joblib.dump(model, "model/model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    print("\nSaved model/model.pkl and model/vectorizer.pkl")


if __name__ == "__main__":
    main()
