import streamlit as st
from keras.models import load_model
from src.model_predicition import predict_tumor, load_and_preprocess_image

# Title
st.title('Brain tumor detection 🧠')
st.write('This is a Streamlit app that uses a Convolutional Neural Network (CNN) to detect brain tumors from MRI images.')

# Header
st.header('🦋 Streamlit app for [Butterfl.ai](http://www.butterfl.ai) interview 🦋')

# Subheader
st.subheader('📚 Instructions 📚')
st.write('1. Upload an MRI image of a brain tumor.')
st.write('2. The app will predict the type of tumor (Glioma, Meningioma, Pituitary, or Normal).')

# File uploader
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
image_path = None

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open(f'/tmp/{uploaded_file.name}', 'wb') as f:
        f.write(uploaded_file.getbuffer())
    image_path = f'/tmp/{uploaded_file.name}'

# Load the trained model
model = load_model('./brain_tumor_cnn.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load image imported ton an array
if uploaded_file is not None and image_path is not None:
    img_array = load_and_preprocess_image(image_path)

    # Predict the tumor type
    predicted_tumor = predict_tumor(model, img_array)
    st.subheader(f'🧠 The predicted tumor type is: {predicted_tumor}')

    # Plot the image with the prediction
    st.image(uploaded_file, use_column_width=True)
