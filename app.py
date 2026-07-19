import streamlit as st
import tensorflow as tf
import pickle
import re
import string
import nltk

from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------
# Download NLTK Data
# ---------------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ---------------------------------
# Load Model
# ---------------------------------
model = tf.keras.models.load_model("fake_news_cnn_model.keras")

# ---------------------------------
# Load Tokenizer
# ---------------------------------
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# ---------------------------------
# Text Cleaning
# ---------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = " ".join(text.split())

    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)

# ---------------------------------
# Prediction Function
# ---------------------------------
def predict_news(news):

    news = clean_text(news)

    sequence = tokenizer.texts_to_sequences([news])

    padded = pad_sequences(
        sequence,
        maxlen=524,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(padded, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        label = "Real News"
        confidence = probability
    else:
        label = "Fake News"
        confidence = 1 - probability

    return label, confidence

# ---------------------------------
# UI
# ---------------------------------

st.title("📰 Fake News Detection Using CNN")

st.markdown(
"""
Paste a news article below and click **Predict**.
The CNN model will determine whether the news is **Real** or **Fake**.
"""
)

news = st.text_area(
    "Enter News Article",
    height=250
)

if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news.")
    else:

        label, confidence = predict_news(news)

        st.divider()

        if label == "Real News":

            st.success("✅ REAL NEWS")

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            st.progress(confidence)

            st.info(
                "The model predicts that this article is likely to be authentic."
            )

        else:

            st.error("❌ FAKE NEWS")

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            st.progress(confidence)

            st.warning(
                "The model predicts that this article is likely to be fake."
            )