## Step 1: import libraries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key, value in word_index.items()}

## Loading model with google drive link
import os
import gdown
from tensorflow.keras.models import load_model

MODEL_PATH = "simple_rnn_imdb.h5"

if not os.path.exists(MODEL_PATH):
    file_id = "1mlXr7-MaNzwtkdVWykz4MGw8uYYv9gvB"
    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False
    )

model = load_model(MODEL_PATH)

## Load the pre-trained model with Relu activation
## model = load_model('simple_rnn_imdb.h5')

## function to decode review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])

## function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=600)
    return padded_review

## Prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

import streamlit as st
## Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive and negative.")

# User input
user_input = st.text_area("Movie Review")

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)
    # make prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
    
else:    
    st.write(f'Please enter a movie review.')