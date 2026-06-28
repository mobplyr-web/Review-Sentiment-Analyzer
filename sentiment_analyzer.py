"""
Review Sentiment Analyzer (Machine Learning)
------------------------------------------------
A beginner/intermediate Machine Learning project that classifies movie
and product reviews as "Positive" or "Negative" using Natural Language
Processing.

How it works:
1. We load a small labeled dataset of example reviews (reviews_dataset.csv).
2. We convert the text into numbers using TF-IDF (Term Frequency -
   Inverse Document Frequency).
3. We train a Logistic Regression classifier on those numbers — a
   simple, very common algorithm for text classification.
4. We can then feed it brand-new reviews and it will predict
   "positive" or "negative".

NOTE: reviews_dataset.csv is a small, synthetic sample so this project
runs instantly with no extra setup. For better real-world accuracy,
swap in a larger public dataset such as the IMDB Movie Reviews dataset
or Amazon product review datasets, then re-train.

Requirements:
    pip install pandas scikit-learn

Run:
    python sentiment_analyzer.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_dataset(path: str = "reviews_dataset.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    print(f"Model accuracy on test data: {accuracy_score(y_test, predictions):.2%}")
    print("\nDetailed report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return model, vectorizer


def predict_review(model, vectorizer, review: str) -> str:
    vec = vectorizer.transform([review])
    prediction = model.predict(vec)[0]
    return prediction.upper()


def main():
    print("Loading dataset and training model...\n")
    df = load_dataset("reviews_dataset.csv")
    model, vectorizer = train_model(df)

    print("\n" + "=" * 50)
    print("Review Sentiment Analyzer - type a review to check it.")
    print("Type 'quit' to exit.")
    print("=" * 50)

    while True:
        review = input("\nEnter a review: ").strip()
        if review.lower() == "quit":
            break
        if not review:
            continue
        result = predict_review(model, vectorizer, review)
        print(f"--> This review sounds: {result}")


if __name__ == "__main__":
    main()
