# Review Sentiment Analyzer (Machine Learning)

A beginner/intermediate Python project that uses NLP + Logistic
Regression to classify movie/product reviews as **positive** or
**negative**.

## What it does
- Loads a labeled dataset of example reviews (`reviews_dataset.csv`).
- Converts the text into numeric features using TF-IDF.
- Trains a Logistic Regression classifier — a simple, widely used
  algorithm for text classification.
- Lets you type in any review and get an instant sentiment prediction.

## Files
- `sentiment_analyzer.py` — main script (vectorizer + model + CLI)
- `reviews_dataset.csv` — small sample dataset (20 positive / 20 negative reviews)

## Requirements
```bash
pip install pandas scikit-learn
```

## How to run
```bash
python sentiment_analyzer.py
```
You'll see the model's accuracy on test data, then you can type any
review to get a live prediction. Type `quit` to exit.

## Notes & next steps
- The included dataset is small so the project runs instantly. For a
  more accurate, real-world model, swap in a larger public dataset such as:
  - [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
  - [Amazon Product Review Datasets](https://nijianmo.github.io/amazon/index.html)
- Ideas to extend this project:
  - Add a third "neutral" class for a 3-way classification problem
  - Try a different model (Naive Bayes, SVM) and compare accuracy
  - Display the top words the model thinks are most positive/negative
