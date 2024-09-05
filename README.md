# Image Recognition Project

This project aims to develop an image recognition system using machine learning techniques. The goal is to detect and classify brain tumors from medical images.

## Installation

1. Clone this repository to your local machine.
2. Make sure you have Python 3.7 (or later) installed.
3. Install the dependencies by running the following command:

```shell
pip install keras h5py keras-resnet
python3 src/install.py
```

### Libraries explanation

- `keras` is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
- `h5py` is a common package to interact with a dataset that is stored on an H5 file.
- `keras-resnet` is a library that provides pre-trained ResNet models.

1. Then enter your kaggle username and your kaggle key to download the dataset.

## Usage

To run the project, execute the following command to train the model:

```shell
python3 src/CNN_model_training.py
```

This will create a model and save it in the `models` directory.

To test the model, execute the following command:

```shell
python3 src/model_prediction.py
```

This will load the model and test it with a sample image.


## Future features

- Implement a feature to detect the tumor location in the image with a bounding box. (YOLO or Faster R-CNN)
- Implement a web interface to upload images and get the prediction.

## Contact

Mail: [Adam Lesage](mailto:adamles44@gmail.com)

LinkedIn: [Adam Lesage](https://www.linkedin.com/in/adam-lesage-341476266/)