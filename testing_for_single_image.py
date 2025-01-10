import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('trained3.h5')

def preprocess_and_predict(image_path):
    img = load_img(image_path, target_size=(224, 224))  
    img = img_to_array(img)  
    img = np.expand_dims(img, axis=0)  
    img = img / 255.0  

    predictions = model.predict(img)

    threshold = 0.5
    predicted_labels = predictions > threshold
    
    return predicted_labels

image_path = 'test/000431.jpg'  

predicted_labels = preprocess_and_predict(image_path)

class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

print("Predicted classes for the image:")
for i, class_name in enumerate(class_names):
    if predicted_labels[0][i]:
        print(f"- {class_name}")

img = load_img(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()
