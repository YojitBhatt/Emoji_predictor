import streamlit as st
import numpy as np
import re
import string
import pickle
import json

# Keras imports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load model
model = load_model("C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\emotion_cnn_model.h5")

# ✅ Load tokenizer from JSON string
with open("C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\tokenizer.json", "r") as f:
    tokenizer_json = f.read()  # Read as string
    tokenizer = tokenizer_from_json(tokenizer_json)  # Convert JSON string to tokenizer

# Load label encoder
with open("C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Emoticon map
emoticon_map = {
    "joy": "😊",
    "sad": "😢",
    "anger": "😠",
    "fear": "😱",
    "love": "❤️",
    "surprise": "😲"
}

# Clean input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"#\w+", "", text)     # Remove hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)      # Remove digits
    text = text.strip()
    return text

# Streamlit UI
st.set_page_config(page_title="Emotion Predictor", page_icon="💬")
st.title("🧠 Emotion Prediction App")
st.write("Enter a sentence to predict its emotion and get an emoticon!")

# User input
user_input = st.text_input("Type your text here:", "")

if user_input:
    cleaned_text = clean_text(user_input)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')  # Match with training maxlen
    pred = model.predict(padded)
    pred_label = np.argmax(pred, axis=1)
    emotion = label_encoder.inverse_transform(pred_label)[0]
    emoticon = emoticon_map.get(emotion, "❓")

    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Emotion:** {emotion}")
    st.markdown(f"**Emoticon:** {emoticon}")
