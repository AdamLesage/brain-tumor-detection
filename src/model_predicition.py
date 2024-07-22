# Imports
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image

# Paths
glioma_tumor_path = './brain-tumors-256x256/Data/glioma_tumor/G_1.jpg'
meningioma_tumor_path = './brain-tumors-256x256/Data/meningioma_tumor/M_1.jpg'
pituitary_tumor_path = './brain-tumors-256x256/Data/pituitary_tumor/P_1.jpg'
no_tumor_path = './brain-tumors-256x256/Data/normal/N_1.jpg'

# Function to load and preprocess a single image
def load_and_preprocess_image(img_path, target_size=(256, 256)):
    """
    Load and preprocess an image.
    :param img_path: Path to the image.
    :param target_size: Target size to resize the image.
    :return: Preprocessed image ready for prediction.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to predict the tumor type
def predict_tumor(model, img_array):
    """
    Predict the tumor type.
    :param model: The trained model.
    :param img_array: Preprocessed image array.
    :return: Predicted tumor type.
    """
    class_names = ['Glioma', 'Meningioma', 'Pituitary', 'Normal']
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return class_names[predicted_class[0]]

# Load the trained model
model = load_model('./models/brain_tumor_cnn.h5')

# Load and preprocess the image
img_array = load_and_preprocess_image(pituitary_tumor_path)

# Predict the tumor type
predicted_tumor = predict_tumor(model, img_array)
print(f"The predicted tumor type is: {predicted_tumor}")

# Plot the image with the prediction
plt.imshow(image.load_img(pituitary_tumor_path))
plt.title(f"Predicted: {predicted_tumor}")
plt.show()
