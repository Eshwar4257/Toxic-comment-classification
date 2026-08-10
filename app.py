import streamlit as st
import pickle
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# Load Model
model = pickle.load(open("toxic_model.pkl", "rb"))

# Load TFIDF
tfidf = pickle.load(open("tfidf.pkl", "rb"))

nltk.download('stopwords')
# Stopwords
stop_words = set(stopwords.words('english'))

# Text Cleaning Function
def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Title
st.title("Toxic Comment Detection System")

st.write(
    "Enter a comment below and click Predict"
)

# User Input
user_text = st.text_area(
    "Enter Comment"
)

# Predict Button
if st.button("Predict"):

    cleaned = clean_text(user_text)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)

    labels = [
        "Toxic",
        "Severe Toxic",
        "Obscene",
        "Threat",
        "Insult",
        "Identity Hate"
    ]

    st.subheader("Prediction Result")

    for i, label in enumerate(labels):

        if prediction[0][i] == 1:

            st.error(f"{label}: YES")

        else:

            st.success(f"{label}: NO")