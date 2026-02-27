import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora
from gensim.models import LdaModel
from wordcloud import WordCloud
from langdetect import detect, DetectorFactory
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import google.generativeai as genai

# ------------------------------
# App Config
# ------------------------------
st.set_page_config(page_title="TeachAIRs", layout="wide")
st.title("📊 TeachAIRs: AI-Powered Student Feedback Analyzer")

# ------------------------------
# API Setup
# ------------------------------

GEMINI_API_KEY = "AIzaSyB-5YS3-Mlj3-AajOj9PwCt4yqRdoTnBHU"
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ------------------------------
# Download NLTK
# ------------------------------
nltk.download("vader_lexicon")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# ------------------------------
# Global Tools
# ------------------------------
lemmatizer = WordNetLemmatizer()
vader = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# ------------------------------
# Preprocessing
# ------------------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# ------------------------------
# Sentiment
# ------------------------------
def get_sentiment(text):
    score = vader.polarity_scores(text)["compound"]
    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return score, label

# ------------------------------
# LDA Topic Modeling
# ------------------------------
def perform_lda(texts, num_topics=4):
    tokenized = [t.split() for t in texts if t]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=3, no_above=0.7)
    corpus = [dictionary.doc2bow(t) for t in tokenized]
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda, dictionary

# ------------------------------
# Gemini Topic Labeling
# ------------------------------
def label_topic(keywords):
    prompt = f"Keywords: {', '.join(keywords)}. Provide a short 3-word topic label."
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("Upload CSV with student feedback", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_column = st.selectbox("Select Feedback Column", df.columns)

    df["Cleaned"] = df[text_column].apply(preprocess)
    df[["Sentiment_Score", "Sentiment_Label"]] = df[text_column].apply(
        lambda x: pd.Series(get_sentiment(x))
    )

    # ------------------------------
    # Sentiment Overview
    # ------------------------------
    st.subheader("📈 Sentiment Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Sentiment_Label", data=df, ax=ax1)
    st.pyplot(fig1)

    # ------------------------------
    # Topic Modeling
    # ------------------------------
    st.subheader("🧠 Topic Modeling")
    num_topics = st.slider("Number of Topics", 2, 8, 4)

    lda_model, dictionary = perform_lda(df["Cleaned"], num_topics)

    for topic_id in range(num_topics):
        words = lda_model.show_topic(topic_id, topn=6)
        keywords = [w for w, _ in words]

        ai_label = label_topic(keywords)

        st.markdown(f"### Topic {topic_id}: {ai_label}")
        st.write("Keywords:", ", ".join(keywords))

        # Wordcloud
        wc_dict = dict(words)
        wc = WordCloud(background_color="white").fit_words(wc_dict)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc)
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    # ------------------------------
    # AI Recommendations
    # ------------------------------
    st.subheader("🤖 AI Teaching Recommendations")

    overall_summary = f"""
    Sentiment Distribution:
    {df['Sentiment_Label'].value_counts().to_dict()}
    """

    if st.button("Generate AI Recommendations"):
        prompt = f"""
        Based on this feedback summary:
        {overall_summary}
        Provide 3 actionable teaching improvement recommendations.
        """
        rec = gemini_model.generate_content(prompt)
        st.success(rec.text)