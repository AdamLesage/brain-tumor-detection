# Imports
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load the data
def load_data(data_dir) -> tuple:
    """
    Load the data from the data directory
    :param data_dir: The directory where the data is stored
    :return: A tuple containing the data and the file names
    """
    data = []
    file_names = []
    classes = os.listdir(data_dir)
    for i in range(len(classes)):
        files = os.listdir(data_dir + '/' + classes[i])
        for file in files:
            img = plt.imread(data_dir + '/' + classes[i] + '/' + file)
            data.append(img)
            file_names.append(file)
    data = np.array(data)
    # Sort the file names in numerical order
    glioma_files = [file for file in file_names if 'G_' in file]
    glioma_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    meningioma_files = [file for file in file_names if 'M_' in file]
    meningioma_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    pituitary_files = [file for file in file_names if 'P_' in file]
    pituitary_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    normal_files = [file for file in file_names if 'N_' in file]
    normal_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    file_names = glioma_files + meningioma_files + pituitary_files + normal_files
    return data, file_names

data, file_names = load_data('brain-tumors-256x256/Data')
len_glioma_data = len([file for file in file_names if 'G_' in file])
glioma_data = data[:len_glioma_data]
x_train_glioma, x_test_glioma, y_train_glioma, y_test_glioma = train_test_split(glioma_data, np.zeros(len_glioma_data), test_size=0.2, random_state=42)
print(f"X_train_glioma shape: {x_train_glioma.shape}")
print(f"X_test_glioma shape: {x_test_glioma.shape}")
print(f"Y_train_glioma shape: {y_train_glioma.shape}")
print(f"Y_test_glioma shape: {y_test_glioma.shape}")

model = Sequential()

# Add the layers to the model
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3))) # 256x256x3 because the images are 256x256 pixels and have 3 channels
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Summary of the model
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_glioma, y_train_glioma, epochs=10, validation_data=(x_test_glioma, y_test_glioma))

# Plot the accuracy and loss
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

score = model.evaluate(x_test_glioma, y_test_glioma, verbose=0)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")

# Save the model
model.save('brain_tumor_cnn.h5')
