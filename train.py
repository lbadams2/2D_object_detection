import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from detect_net import DetectNet
import params
from create_dataset import draw_orig_image

# only penalize classification grid cell using SSE
# only penalize coordinate loss if object present in grid cell and box responsible for that object
# for cells with no object penalize classification score using SSE
# for boxes in grid cell that aren't responsible for object (only 1 if 2 anchor boxes) do SSE on object confidence score
# sum losses for all grid cells
def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_, _ = model(x, training=training)
    total_loss = 0

    for i in range(params.batch_size):
        for j in range(y_.shape[1]): # rows
            for k in range(y_.shape[2]): # columns
                objs_in_cell = []
                for obj in y[i]: # get true objects in current grid cell
                    cell_x = obj[0] // params.grid_stride
                    cell_y = obj[1] // params.grid_stride
                    if cell_x.numpy() == j and cell_y.numpy() == k:
                        objs_in_cell.append(obj)

                if objs_in_cell:
                    highest_iou = -1
                    num_objs_cell = len(objs_in_cell)
                    # iou_arr contains the ious between each box in cell and all objects in cell
                    iou_arr = tf.zeros([num_objs_cell, params.num_anchors], tf.float32)
                    iou_arr = tf.Variable(iou_arr)
                    for b in range(params.num_anchors): # find predicted box with highest iou with obj in current cell
                        start_ind = b * params.vec_len
                        end_ind = start_ind + params.vec_len
                        pred_vec = y_[i, j, k, start_ind:end_ind]
                        ious = DetectNet.bbox_iou(pred_vec, objs_in_cell) # iou of b with all objs
                        iou_arr[:,b].assign(ious)

                    # while list size < num_boxes, get max over all rows, if row index of max not in list append to list, make max 0 in iou_arr
                    # pick unique bounding box for each true object
                    pred_to_obj = {}
                    obj_to_pred = {}
                    while len(pred_to_obj) < min(params.num_anchors, num_objs_cell):
                        max_ = tf.math.reduce_max(iou_arr) # returns max val
                        max_mask = tf.equal(iou_arr, max_)
                        max_loc = tf.where(max_mask)[0] # just get first max
                        max_obj, max_b = max_loc[0].numpy(), max_loc[1].numpy()
                        if max_b not in pred_to_obj and max_obj not in obj_to_pred:
                            pred_to_obj[max_b] = max_obj
                            obj_to_pred[max_obj] = max_b
                            bc = tf.constant(-1, shape=[num_objs_cell], dtype=tf.float32)
                            iou_arr[:,max_b].assign(bc) # so this box won't be selected again

                    # if cell contains object(s), only calculate coordinate and class prob loss for box resp for each object
                    for b in pred_to_obj:
                        start_ind = b * params.vec_len
                        end_ind = start_ind + params.vec_len
                        pred_vec = y_[i, j, k, start_ind:end_ind]
                        true_vec = y[i, pred_to_obj[b]]
                        # need to ensure coordinates for pred vec are absolute over entire image
                        center_sse = (true_vec[0] - pred_vec[0])**2 + (true_vec[1] - pred_vec[1])**2
                        total_loss += params.coord_loss_weight * center_sse

                        wh_sse = (true_vec[2] - pred_vec[2])**2 + (true_vec[3] - pred_vec[3])**2
                        total_loss = params.coord_loss_weight * wh_sse

                        c_idx = 5
                        class_sse = 0
                        # YOLO paper only does 1 set of class probs per cell, not per bounding box
                        # may need to adjust this
                        for c in range(c_idx, c_idx + params.num_classes):
                            class_sse += (true_vec[c] - pred_vec[c])**2
                        total_loss += class_sse

                        obj_conf_sse = (1 - pred_vec[4])**2
                        total_loss += obj_conf_sse

                    # can also do classification error for box not used if there is one
                    # YOLO paper only does 1 set of class probs per cell

                # if no object in cell, only calculate loss on the object confidence for each box in cell
                else:
                    for b in range(params.num_anchors):
                        start_ind = b * params.vec_len
                        end_ind = start_ind + params.vec_len
                        pred_vec = y_[i, j, k, start_ind:end_ind]
                        obj_conf_sse = (1 - pred_vec[4])**2
                        obj_conf_sse = params.noobj_loss_weight * obj_conf_sse
                        total_loss += obj_conf_sse

    return total_loss


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def comp_nms_gt(nms_boxes, gt_boxes):
    pass


def _parse_image_function(example):
    # Create a dictionary describing the features.
    context_feature = {
        'image': tf.io.FixedLenFeature([], dtype=tf.string)
    }
    sequence_features = {
        'Box Vectors': tf.io.VarLenFeature(dtype=tf.float32)
    }
    context_data, sequence_data = tf.io.parse_single_sequence_example(serialized=example, 
                                    context_features=context_feature, sequence_features=sequence_features)

    return context_data['image'], sequence_data['Box Vectors']


def format_data(image, labels):
    vecs = tf.sparse.to_dense(labels)
    #print(vecs)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # this should also normalize pixels
    #print(image)
    return image, vecs

# each object in training image is assigned to grid cell that contains object's midpoint
# and anchor box for the grid cell with highest IOU
def train():
    FILENAME = 'image_dataset_train.tfrecord'
    dataset = tf.data.TFRecordDataset(FILENAME)
    dataset = dataset.map(_parse_image_function)
    dataset = dataset.map(format_data)
    dataset = dataset.padded_batch(params.batch_size, padded_shapes=([None, None, 3], [None, None]))

    '''
    VAL_FILENAME = 'data/validation/segment-272435602399417322_2884_130_2904_130_with_camera_labels.tfrecord'
    val_dataset = tf.data.TFRecordDataset(VAL_FILENAME, compression_type='')
    val_dataset.batch(params.batch_size)
    total_val = len(val_dataset)
    '''

    model = DetectNet(True)
    #model.compile(optimizer='adam',
    #              loss = 'mse',
    #              metrics=['accuracy'])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # to get validation stats need to do nms during training also, return result of nms in addition to all boxes
    # then check iou of each nms box with each ground truth from val set, if above threshold compare classification, use comp_nms_gt()
    #for epoch in range(params.epochs):
    for x, y in dataset:
        #print(x)
        #print(y)
        loss_value, grads = grad(model, x, y)
        print('train loss is ', loss_value)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


# don't call grad and loss here, just get nms from model and pass to comp_nms_gt()
def test():
    pass


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