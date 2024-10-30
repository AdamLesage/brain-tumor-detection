import streamlit as st
from keras.models import load_model
from src.model_predicition import predict_tumor, load_and_preprocess_image

# Title
st.title('Brain tumor detection 🧠')
st.write('This is a Streamlit app that uses a Convolutional Neural Network (CNN) to detect brain tumors from MRI images.')

# Header
st.header('🦋 Streamlit app for [Butterfl.ai](http://www.butterfl.ai) interview 🦋')

st.write('Not available for Streamlit sharing due to the model file size.')
st.write('Contact me at [Adam Lesage email](mailto:adamles44@gmail.com) for more information.')