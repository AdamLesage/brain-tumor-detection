# Imports
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


# Ask to the model to make predictions from an image given in input
def predict_image(image_path: str, model_path: str) -> str:
    """
    Predict the class of an image
    :param image_path: The path of the image
    :return: The class of the image
    """
    model = load_model(model_path)
    img = plt.imread(image_path)
    img = np.array([img])
    prediction = model.predict(img)
    response = ''
    if prediction[0][0] == 1:
        response = 'Glioma Tumor'
    elif prediction[0][1] == 1:
        response = 'Meningioma Tumor'
    elif prediction[0][2] == 1:
        response = 'Pituitary Tumor'
    elif prediction[0][3] == 1:
        response = 'No Tumor'
    return response

glioma_tumor_path = './brain-tumors-256x256/Data/glioma_tumor/P_1.jpg'
meningioma_tumor_path = './brain-tumors-256x256/Data/meningioma_tumor/M_1.jpg'
pituitary_tumor_path = './brain-tumors-256x256/Data/pituitary_tumor/P_1.jpg'
no_tumor_path = './brain-tumors-256x256/Data/no_tumor/N_1.jpg'

print(predict_image(image_path=meningioma_tumor_path, model_path='./models/brain_tumor_cnn.h5'))