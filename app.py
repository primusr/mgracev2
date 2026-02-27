import streamlit as st
import pandas as pd

from utils.preprocessing import preprocess_text
from utils.sentiment import analyze_sentiment
from utils.topic_modeling import perform_lda
from utils.gemini_ai import label_topic, generate_recommendations
from utils.visualizations import (
    plot_sentiment_distribution,
    generate_wordcloud
)

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(page_title="TeachAIRs", layout="wide")
st.title("📊 TeachAIRs: AI-Powered Student Feedback Analyzer")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV with student feedback", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_column = st.selectbox("Select Feedback Column", df.columns)

    # -----------------------------
    # Preprocessing
    # -----------------------------
    df["Cleaned"] = df[text_column].apply(preprocess_text)

    # -----------------------------
    # Sentiment Analysis
    # -----------------------------
    df[["Sentiment_Score", "Sentiment_Label"]] = df[text_column].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )

    st.subheader("📈 Sentiment Distribution")
    fig = plot_sentiment_distribution(df)
    st.pyplot(fig)

    # -----------------------------
    # Topic Modeling
    # -----------------------------
    st.subheader("🧠 Topic Modeling")
    num_topics = st.slider("Number of Topics", 2, 8, 4)

    lda_model, dictionary = perform_lda(df["Cleaned"], num_topics)

    for topic_id in range(num_topics):
        words = lda_model.show_topic(topic_id, topn=6)
        keywords = [w for w, _ in words]

        topic_label = label_topic(keywords)

        st.markdown(f"### Topic {topic_id + 1}: {topic_label}")
        st.write("Keywords:", ", ".join(keywords))

        wc_fig = generate_wordcloud(words)
        st.pyplot(wc_fig)

    # -----------------------------
    # AI Recommendations
    # -----------------------------
    st.subheader("🤖 AI Teaching Recommendations")

    if st.button("Generate AI Recommendations"):
        summary = df["Sentiment_Label"].value_counts().to_dict()
        recommendations = generate_recommendations(summary)
        st.success(recommendations)