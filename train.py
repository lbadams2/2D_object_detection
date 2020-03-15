from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import tensorflow as tf
import matplotlib.pyplot as plt

from detect_net import DetectNet
import params

def IOU(box1, box2):
    x_c, y_c, w1, h1 = box1[0], box1[1], box1[2], box1[3]
    x2_c, y2_c, w2, h2 = box2[0], box2[1], box2[2], box2[3]
    
    # get bottom left coordinates
    x1 = x_c - w1 / 2
    y1 = y_c - h1 / 2
    x2 = x2_c - w2 / 2
    y2 = y2_c - h2 / 2

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)

    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U


def read_tfrecord(example):
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(example.numpy()))
    # each frame consists of 5 different camera images
    #front_image = None
    #for index, image in enumerate(frame.images):
    #    if image.name == 1: # 1 is front image
    #        front_image = image
    #    break
    front_labels = None
    front_image = None
    for camera_labels in frame.camera_labels:
        if camera_labels.name != 1: # only take front camera images
            continue
        else:
            front_labels = camera_labels
            break
    for image in frame.images:
        if image.name != 1:
            continue
        else:
            front_image = image

    num_objects = len(front_labels.labels)
    true_boxes = np.zeros(num_objects, 7)
    for i in range(num_objects):
        label = front_labels.labels[i]
        box_vector = np.zeros((7, 1))
        box_vector[0] = label.box.center_x
        box_vector[1] = label.box.center_y
        box_vector[2] = label.box.width
        box_vector[3] = label.box.length
        if label.type == 'TYPE_PEDESTRIAN':
            box_vector[4] = 1
        elif label.type == 'TYPE_VEHICLE':
            box_vector[5] = 1
        else:
            box_vector[6] = 1
        true_boxes[i] = box_vector
    
    #train_boxes = create_boxes()
    return true_boxes, front_image



def print_results(model, dataset):
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


def loss(model, x, y, training, loss_object):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
        y_ = model(x, training=training)

        return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train():
    FILENAME = 'data/train/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    dataset.batch(batch_size)
    total_train = len(dataset)
    num_imgs = total_train

    VAL_FILENAME = 'data/validation/segment-272435602399417322_2884_130_2904_130_with_camera_labels.tfrecord'
    val_dataset = tf.data.TFRecordDataset(VAL_FILENAME, compression_type='')
    val_dataset.batch(batch_size)
    total_val = len(val_dataset)
    
    model = DetectNet(im_width, im_height)
    #model.compile(optimizer='adam',
    #              loss = 'mse',
    #              metrics=['accuracy'])
    loss_object = tf.keras.losses.MeanSquaredError()
    

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    for epoch in range(epochs):
        for x, y, img in dataset:
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

if __name__ == '__main__':
    train()