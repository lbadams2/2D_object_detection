import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

########## Hyper Parameters ##################
batch_size = 16
epochs = 10
boxes_per_img = 8
##############################################

im_height = 300
im_width = 300

# don't have to actually draw on image, create [num_imgs x num_box_per_img x x_center x y_center] array
def create_boxes():
    bboxes = np.zeros((num_imgs, boxes_per_img, 4)) # learn the 4 coordinates (x_center, y_center, x_len, y_len) for each box in each image
    box_height = im_height / 8
    box_width = im_width / 8
    for i_img in range(num_imgs):
        for x in range(boxes_per_img):
            for y in range(boxes_per_img):         
                (left, right, top, bottom) = (x * box_width, (x + 1) * box_width, y * box_height, (y + 1) * box_height)
                x_c = (left + right) / 2
                y_c = (top + bottom) / 2
                bboxes[i_img, x * y] = [x_c, y_c, box_width, box_height]

'''
def add_bounding_boxes(image):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    box_height = im_height / 8
    box_width = im_width / 8
    for w in range(8):
        for h in range(8):        
            (left, right, top, bottom) = (w * box_width, (w + 1) * box_width, h * box_height, (h + 1) * box_height)
            draw.line([(left, top), (left, bottom), (right, bottom), (right, top),(left, top)])
'''

def read_tfrecord(example):
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(example.numpy()))
    # each frame consists of 5 different camera images
    front_image = None
    for index, image in enumerate(frame.images):
        if image.name == 'front':
            front_image = image
        break
    #add_bounding_boxes(front_image.image)
    #image = tf.cast(image, tf.float32) / 255.0
    #image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])
    
    #return image, class_label
    return front_image



FILENAME = 'data/train/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
dataset.map(read_tfrecord, num_parallel_calls=AUTO)
dataset.batch(batch_size)
total_train = len(dataset)

VAL_FILENAME = 'data/validation/segment-272435602399417322_2884_130_2904_130_with_camera_labels.tfrecord'
val_dataset = tf.data.TFRecordDataset(VAL_FILENAME, compression_type='')
val_dataset.batch(batch_size)
total_val = len(val_dataset)

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(7) # 4 coordinates and 3 dim 1 hot vector for object class
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CrossEntropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit_generator(
    dataset,
    steps_per_epoch=total_train, # batch_size,
    epochs=epochs,
    validation_data=val_dataset,
    validation_steps=total_val # batch_size
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()