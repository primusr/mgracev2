import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")
vader = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = vader.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return score, label