# Image Recognition using MobileNetV2 with VOC2007 Dataset
This project demonstrates the application of deep learning techniques for multi-label image classification using the VOC2007 dataset. The model is based on the MobileNetV2 architecture, which is fine-tuned to classify images into 20 object categories. Each image can contain multiple objects, making this a multi-label classification problem.

## Project Overview

The goal of this project is to develop a deep learning model that can predict multiple object classes in an image from the VOC2007 dataset. The dataset includes 20 different object categories such as aeroplane, bicycle, dog, person, etc. The model uses transfer learning with MobileNetV2 and fine-tunes it on the VOC2007 dataset to achieve accurate predictions.

## Features

- Multi-label image classification.
- Utilizes pre-trained MobileNetV2 for feature extraction.
- Data augmentation techniques to enhance model generalization.
- Evaluation using precision, recall, and F1-score.
- Visualizations of loss and accuracy during training.

## Installation

To run this project, you need to have Python 3.6+ installed, along with the following dependencies:

- TensorFlow 2.x
- Numpy
- Scikit-learn
- Matplotlib
- PIL (Pillow)

### Install dependencies

```bash
pip install tensorflow numpy scikit-learn matplotlib Pillow
```

### How to work with this model?

#### Cloning the repository

```bash
git clone https://github.com/saksham1253/Image-Recognition-using-MobileNetV2.git
```

#### Where to find the file?

Wherever you run the terminal presently in which folder it is open in that folder a clone of the model is made.

#### Training the model

![Defining the Path of Dataset for training the model]("pathtrain.png")

 



