import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import base64
import zipfile
import os
import tempfile
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import docx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import uuid

nltk.download('stopwords')
nltk.download('vader_lexicon')

st.set_page_config(layout="wide")
st.title("🌐 Advanced Word Cloud Generator")

# Initialize session state
if "clouds" not in st.session_state:
    st.session_state.clouds = []

# Built-in shape masks
shape_library = {
    "None": None,
    "Circle": "circle.png",
    "Star": "star.png",
    "Heart": "heart.png"
}

# Load shape masks
def load_mask(shape_name):
    if shape_name == "None":
        return None
    path = os.path.join("shapes", shape_library[shape_name])
    return np.array(Image.open(path).convert("L"))

# Extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except:
        return ""

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Generate word cloud
def generate_wordcloud(text, mask, font_path, colormap, bg_color, stopword_toggle):
    stop_words = set(stopwords.words("english")) if stopword_toggle else set()
    wc = WordCloud(width=800, height=400, background_color=bg_color,
                   mask=mask, font_path=font_path, colormap=colormap,
                   stopwords=stop_words).generate(text)
    return wc

# Generate word frequency chart
def plot_word_freq(text, stopword_toggle):
    stop_words = set(stopwords.words("english")) if stopword_toggle else set()
    words = [w.lower() for w in text.split() if w.lower() not in stop_words]
    freq = pd.Series(words).value_counts().head(20)
    fig, ax = plt.subplots()
    sns.barplot(x=freq.values, y=freq.index, ax=ax)
    ax.set_title("Top 20 Words")
    return fig

# Sentiment analysis
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

# Topic modeling
def extract_topics(text, n_topics=3):
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    topics = []
    for topic in lda.components_:
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]
        topics.append(", ".join(top_words))
    return topics

# Keyword extraction
def extract_keywords(text, n=10):
    words = [w.lower() for w in text.split()]
    freq = pd.Series(words).value_counts()
    return freq.head(n).index.tolist()

# Text summarization (simple)
def summarize_text(text, n=3):
    sentences = text.split(".")
    return ". ".join(sentences[:n]) + "."

# Save image to buffer
def save_image(wc):
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return buf

# Save all clouds to zip
def save_all_clouds(clouds):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for i, cloud in enumerate(clouds):
            img_buf = save_image(cloud["wordcloud"])
            zf.writestr(f"wordcloud_{i+1}.png", img_buf.read())
    zip_buf.seek(0)
    return zip_buf

# Main UI
st.sidebar.header("➕ Add New Word Cloud")
text_input = st.sidebar.text_area("Enter text or paste URL", height=150)
upload_file = st.sidebar.file_uploader("Or upload a file (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
url_mode = st.sidebar.checkbox("Treat input as URL")
shape = st.sidebar.selectbox("Choose shape", list(shape_library.keys()))
font_file = st.sidebar.file_uploader("Upload custom font (.ttf)", type=["ttf"])
colormap = st.sidebar.selectbox("Color theme", plt.colormaps())
bg_color = st.sidebar.color_picker("Background color", "#ffffff")
stopword_toggle = st.sidebar.checkbox("Remove stopwords", value=True)

if st.sidebar.button("Generate Word Cloud"):
    if url_mode:
        text = extract_text_from_url(text_input)
    elif upload_file:
        if upload_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(upload_file)
        elif upload_file.name.endswith(".docx"):
            text = extract_text_from_docx(upload_file)
        else:
            text = upload_file.read().decode("utf-8")
    else:
        text = text_input

    if text.strip():
        mask = load_mask(shape)
        font_path = None
        if font_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ttf") as tmp:
                tmp.write(font_file.read())
                font_path = tmp.name
        wc = generate_wordcloud(text, mask, font_path, colormap, bg_color, stopword_toggle)
        st.session_state.clouds.append({
            "text": text,
            "wordcloud": wc,
            "keywords": extract_keywords(text),
            "summary": summarize_text(text),
            "sentiment": analyze_sentiment(text),
            "topics": extract_topics(text)
        })

# Display all word clouds
for i, cloud in enumerate(st.session_state.clouds):
    st.subheader(f"Word Cloud {i+1}")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(cloud["wordcloud"].to_image())
        st.pyplot(plot_word_freq(cloud["text"], stopword_toggle))
    with col2:
        st.markdown("**Summary:**")
        st.write(cloud["summary"])
        st.markdown("**Keywords:**")
        st.write(", ".join(cloud["keywords"]))
        st.markdown("**Topics:**")
        st.write(cloud["topics"])
        st.markdown("**Sentiment:**")
        st.json(cloud["sentiment"])
        st.download_button("📥 Download Image", save_image(cloud["wordcloud"]), file_name=f"wordcloud_{i+1}.png")
        st.download_button("📄 Download Word List", data="\n".join(cloud["keywords"]), file_name=f"words_{i+1}.txt")

# Download all as zip
if st.session_state.clouds:
    zip_buf = save_all_clouds(st.session_state.clouds)
    st.download_button("📦 Download All Word Clouds as ZIP", zip_buf, file_name="all_wordclouds.zip")
