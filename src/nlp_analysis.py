import pandas as pd
from transformers import pipeline

def analyze_sentiment(texts):
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiments = [sentiment_pipeline(text)[0] for text in texts]
    return sentiments

def load_open_responses(filepath):
    df = pd.read_csv(filepath)
    # Check if 'comments' column exists (from original data)
    if 'comments' in df.columns:
        return df['comments'].dropna().tolist()
    elif 'open_response' in df.columns:
        return df['open_response'].dropna().tolist()
    else:
        print("No text column found for NLP analysis. Available columns:", list(df.columns))
        return []

if __name__ == "__main__":
    responses = load_open_responses('../Data/processed/processed_data.csv')
    if responses:
        sentiments = analyze_sentiment(responses[:5])  # Sample first 5
        print(sentiments)
    else:
        print("No text responses found for sentiment analysis.")