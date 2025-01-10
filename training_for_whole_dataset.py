import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import xml.etree.ElementTree as ET
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

train_dir = 'VOC2007/JPEGImages'
train_list_file = 'VOC2007/ImageSets/Main/train.txt'
val_list_file = 'VOC2007/ImageSets/Main/val.txt'
annotations_dir = 'VOC2007/Annotations'

class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

def parse_annotation(annotation_file):
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    label = np.zeros(20)
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = class_names.index(class_name)
        label[class_id] = 1
    return label

def create_image_generator(image_list_file, image_dir, annotations_dir, target_size=(224, 224), batch_size=32):
    with open(image_list_file, 'r') as file:
        image_list = file.read().splitlines()

    while True:
        batch_images = []
        batch_labels = []

        for image_name in image_list:
            img_path = os.path.join(image_dir, image_name + '.jpg')
            annotation_path = os.path.join(annotations_dir, image_name + '.xml')

            if not os.path.exists(img_path) or not os.path.exists(annotation_path):
                continue

            img = load_img(img_path, target_size=target_size)
            img = img_to_array(img)
            label = parse_annotation(annotation_path)

            batch_images.append(img)
            batch_labels.append(label)

            if len(batch_images) >= batch_size:
                yield np.array(batch_images), np.array(batch_labels)
                batch_images = []
                batch_labels = []

train_generator = create_image_generator(train_list_file, train_dir, annotations_dir, batch_size=32)
val_generator = create_image_generator(val_list_file, train_dir, annotations_dir, batch_size=32)

base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(20, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=156,
    epochs=10,
    validation_data=val_generator,
    validation_steps=155,
    batch_size=32
)

model.save('voc2007_model_finetuned.h5')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss during Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy during Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

y_true = []
y_pred = []

for batch_images, batch_labels in val_generator:
    y_true.extend(batch_labels)
    y_pred.extend(model.predict(batch_images))

y_true = np.array(y_true)
y_pred = (np.array(y_pred) > 0.5).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
