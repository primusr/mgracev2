import streamlit as st
import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

def label_topic(keywords):
    prompt = f"""
    Keywords: {', '.join(keywords)}
    Provide a short 3-word academic topic label.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_recommendations(summary_dict):
    prompt = f"""
    Based on this student feedback sentiment distribution:
    {summary_dict}

    Provide 3 actionable teaching improvement recommendations.
    """
    response = model.generate_content(prompt)
    return response.text.strip()