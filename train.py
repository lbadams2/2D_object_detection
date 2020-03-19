from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import tensorflow as tf
import numpy as np
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
        box_vector = np.zeros((8, 1))
        box_vector[0] = label.box.center_x
        box_vector[1] = label.box.center_y
        box_vector[2] = label.box.width
        box_vector[3] = label.box.length
        box_vector[4] = 1 # objectness score
        if label.type == 'TYPE_PEDESTRIAN':
            box_vector[5] = 1
        elif label.type == 'TYPE_VEHICLE':
            box_vector[6] = 1
        else:
            box_vector[7] = 1
        true_boxes[i] = box_vector
    
    #train_boxes = create_boxes()
    return true_boxes, front_image


# only penalize classification grid cell using SSE
# only penalize coordinate loss if object present in grid cell and box responsible for that object
# for cells with no object penalize classification score using SSE
# for boxes in grid cell that aren't responsible for object (only 1 if 2 anchor boxes) do SSE on object confidence score
# sum losses for all grid cells
def loss(model, x, y, training, loss_object):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_, _ = model(x, training=training)
    num_boxes_cell = y_.shape()[2]
    total_loss = 0
    for i in range(y_.shape()[0]): # rows
        for j in range(y_.shape()[1]): # columns
            objs_in_cell = []
            idxs = []
            idx = 0
            for obj in y: # get true objects in current grid cell
                cell_x = obj[0] / params.stride
                cell_y = obj[1] / params.stride
                if cell_x == i and cell_y == j:
                    objs_in_cell.append(obj)
                    idxs.append(idx)
                idx += 1

            if objs_in_cell:
                highest_iou = -1
                num_objs_cell = len(objs_in_cell)
                iou_arr = np.zeros(1, num_objs_cell)
                for b in range(num_boxes_cell): # find predicted box with highest iou with obj in current cell
                    pred_vec = y_[i, j, b]
                    ious = DetectNet.bbox_iou(pred_vec, objs_in_cell) # iou of b with all objs
                    if iou_arr.shape()[0] == 1:
                        iou_arr[0] = ious
                    else:
                        iou_arr = np.vstack([iou_arr, ious]) # make ious into row if not already

                # while list size < num_boxes, get max over all rows, if row index of max not in list append to list, make max 0 in iou_arr
                # pick unique bounding box for each true object
                pred_to_obj = {}
                obj_to_pred = {}
                while len(pred_to_obj) < min(num_boxes_cell, num_objs_cell):
                    max_ = np.amax(iou_arr)
                    result = numpy.where(iou_arr == max_)
                    cord_list = list(zip(result[0], result[1]))
                    for cord in cord_list:
                        if cord[0] not in pred_to_obj and cord[1] not in obj_to_pred:
                            pred_to_obj[cord[0]] = cord[1]
                            obj_to_pred[cord[1]] = cord[0]
                            iou_arr[cord[0], cord[1]] = -1

                # if cell contains object(s), only calculate coordinate and class prob loss for box resp for each object
                for b in pred_to_obj:
                    pred_vec = y_[i, j, b]
                    true_vec = y[pred_to_obj[b]]
                    # need to ensure coordinates for pred vec are absolute over entire image
                    center_sse = (true_vec[0] - pred_vec[0])^2 + (true_vec[1] - pred_vec[1])^2
                    total_loss += params.coord_loss_weight * center_sse
                    
                    wh_sse = (true_vec[2] - pred_vec[2])^2 + (true_vec[3] - pred_vec[3])^2
                    total_loss = params.coord_loss_weight * wh_sse

                    c_idx = 5
                    class_sse = 0
                    # YOLO paper only does 1 set of class probs per cell, not per bounding box
                    # may need to adjust this
                    for c in range(params.num_classes):
                        class_sse += (true_vec[c_idx + c] - pred_vec[c_idx + c])^2
                    total_loss += class_sse

                    obj_conf_sse = (1 - pred_vec[4])^2 + (1 - pred_vec[4])^2
                    obj_conf_sse = params.noobj_loss_weight * obj_conf_sse
                    total_loss += obj_conf_sse

            # if no object in cell, only calculate loss on the object confidence for each box in cell
            else:
                for b in range(num_boxes_cell):
                    pred_vec = y_[i, j, b]
                    obj_conf_sse = (1 - pred_vec[4])^2 + (1 - pred_vec[4])^2
                    obj_conf_sse = params.noobj_loss_weight * obj_conf_sse
                    total_loss += obj_conf_sse

    return total_loss


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# each object in training image is assigned to grid cell that contains object's midpoint
# and anchor box for the grid cell with highest IOU
def train():
    FILENAME = 'data/train/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    dataset.batch(params.batch_size)
    total_train = len(dataset)
    num_imgs = total_train

    VAL_FILENAME = 'data/validation/segment-272435602399417322_2884_130_2904_130_with_camera_labels.tfrecord'
    val_dataset = tf.data.TFRecordDataset(VAL_FILENAME, compression_type='')
    val_dataset.batch(params.batch_size)
    total_val = len(val_dataset)
    
    model = DetectNet(params.im_width, params.im_height)
    #model.compile(optimizer='adam',
    #              loss = 'mse',
    #              metrics=['accuracy'])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for epoch in range(params.epochs):
        for x, y, img in dataset:
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))


def print_results(model, dataset):
    history = model.fit_generator(
        dataset,
        steps_per_epoch=total_train, # batch_size,
        epochs=params.epochs,
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

if __name__ == '__main__':
    train()