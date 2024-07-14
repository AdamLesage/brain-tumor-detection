# Imports
import tensorflow as tf


model = tf.keras.applications.MobileNetV2(weights='imagenet')
img_path = "brain-tumors-256x256/Data/glioma_tumor/G_1.jpg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
input_img = tf.keras.preprocessing.image.img_to_array(img)
input_img = tf.keras.applications.mobilenet.preprocess_input(input_img)
input_img = tf.expand_dims(input_img, axis=0)

predictions = model.predict(input_img)
predicted_classes = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]
for class_id, class_name, class_likelihood in predicted_classes:
    print(f"Class: {class_name}, Likelihood: {class_likelihood}")