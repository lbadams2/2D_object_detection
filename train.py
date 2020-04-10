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
def loss(model, x, true_box_grid, true_box_mask, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_, _ = model(x, training=training)
    # (batch, rows, cols, anchors, vals)
    center_coords, wh_coords, obj_scores, class_probs = DetectNet.predict_transform(y_)
    total_loss = 0

    pred_wh_half = wh_coords / 2.
    # bottom left corner
    pred_mins = center_coords - pred_wh_half
    # top right corner
    pred_maxes = center_coords + pred_wh_half

    true_xy = true_box_grid[..., 0:2]
    true_wh = true_box_grid[..., 2:4]
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    # max bottom left corner
    intersect_mins = tf.math.maximum(pred_mins, true_mins)
    # min top right corner
    intersect_maxes = tf.math.minimum(pred_maxes, true_maxes)    
    intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins, 0.)
    # product of difference between x max and x min, y max and y min
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]


    pred_areas = wh_coords[..., 0] * wh_coords[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    # Best IOUs for each location.
    iou_scores = tf.expand_dims(iou_scores, 4)
    best_ious = tf.keras.backend.max(iou_scores, axis=4)  # Best IOU scores.
    best_ious = tf.expand_dims(best_ious, 4)

    # A detector has found an object if IOU > thresh for some true box.
    object_detections = tf.keras.backend.cast(best_ious > 0.6, dtype=tf.float32)

    # for both no obj and obj only calculate loss for boxes that had high ious
    no_obj_weights = params.noobj_loss_weight * (1 - object_detections) * (1 - true_box_mask[...,:1])
    no_obj_loss = no_obj_weights * tf.math.square(obj_scores)

    obj_conf_loss = params.coord_loss_weight * true_box_mask[...,:1] * tf.math.square(1 - obj_scores)
    conf_loss = no_obj_loss + obj_conf_loss
    class_loss = true_box_mask * tf.math.square(true_box_grid[..., 5:] - class_probs)

    # keras_yolo does a sigmoid on center_coords here but they should already be between 0 and 1 from predict_transform
    pred_boxes = tf.concat([center_coords, wh_coords], axis=-1)
    coord_loss = true_box_mask[...,:4] * tf.math.square(true_box_grid[...,:4] - pred_boxes)

    confidence_loss_sum = tf.keras.backend.sum(conf_loss)
    classification_loss_sum = tf.keras.backend.sum(class_loss)
    coordinates_loss_sum = tf.keras.backend.sum(coord_loss)

    # not sure why .5 is here, maybe to make sure numbers don't get too large
    total_loss = 0.5 * (confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)            

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
    image = tf.image.resize(image, [params.scaled_height, params.scaled_width])
    #print(image)
    return image, vecs


# tensors are (rows, cols)
# coords are (x, y) - opposite
# create class mask here too
def create_true_box_grid(y):
    orig_size = tf.keras.backend.stack((params.im_width, params.im_height))
    orig_size = tf.cast(orig_size, tf.float32)
    center_boxes = y[:,:2] / orig_size
    #center_boxes = tf.cast(center_boxes, tf.float32)
    boxes_wh = y[:,2:4] / orig_size
    #boxes_wh = tf.cast(boxes_wh, tf.float32)
    conv_size = tf.keras.backend.stack((params.grid_width, params.grid_height))
    conv_size = tf.cast(conv_size, tf.float32)
    center_boxes = center_boxes * conv_size
    boxes_wh = boxes_wh * conv_size
    # box coords are now normalized to be between [grid_width, grid_height] (same scale as anchors)
    true_boxes = tf.concat([center_boxes, boxes_wh, y[:,4:]], axis=1)

    anchors = DetectNet.create_anchors().numpy()
    true_box_grid = tf.zeros(shape=[params.grid_height, params.grid_width, params.num_anchors, params.vec_len], dtype=tf.float32)

    box_mask = tf.zeros(shape=[params.grid_height, params.grid_width, params.num_anchors, 5], dtype=tf.float32)
    mask_vec = tf.ones([5], dtype=tf.float32)
    
    num_objs = y.shape[0]
    for obj in range(num_objs):
        best_iou = 0
        best_anchor = 0
        box = true_boxes[obj]
        i = int(tf.math.floor(box[1]).numpy()) # row
        #i = tf.cast(i, tf.float32)
        j = int(tf.math.floor(box[0]).numpy()) # column
        #j = tf.cast(j, tf.float32) 

        # find best anchor for current box
        for k, anchor in enumerate(anchors):
            # center box
            box_max = box[2:4] / 2
            box_min = -box_max

            # center anchor
            anchor_max = anchor / 2
            anchor_min = -anchor_max

            intersect_mins = tf.math.maximum(box_min, anchor_min)
            intersect_maxes = tf.math.minimum(box_max, anchor_max)
            intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        # its possible 2 objects in same cell have same anchor, last will overwrite
        if best_iou > 0:
            box_vec = tf.constant(0, shape=[params.vec_len], dtype=tf.float32)
            box_vec = tf.Variable(box_vec)
            box_vec[0].assign(box[0] - i) # center should be between 0 and 1, like prediction will be
            box_vec[1].assign(box[1] - j) # center should be between 0 and 1, like prediction will be
            adjusted_width = tf.math.log(box[2] / anchors[best_anchor][0]) # quotient might be less than one, not sure why log is used
            box_vec[2].assign(adjusted_width)
            adjusted_height = tf.math.log(box[3] / anchors[best_anchor][1]) # quotient might be less than one, not sure why log is used
            box_vec[3].assign(adjusted_height)
            true_box_grid[i, j, best_anchor, :].assign(box_vec)
            box_mask[i, j, best_anchor, :].assign(mask_vec)
        
    # could pad to 1920 x 1920 with 0s then resize to 416 x 416
    
    return true_box_grid, box_mask
        

# each object in training image is assigned to grid cell that contains object's midpoint
# and anchor box for the grid cell with highest IOU
def train():
    FILENAME = 'image_dataset_train.tfrecord'
    dataset = tf.data.TFRecordDataset(FILENAME)    
    dataset = dataset.map(_parse_image_function)
    dataset = dataset.map(format_data)

    
    print('Begin data preprocessing')
    true_box_grids = []
    true_box_masks = []
    for _, y in dataset:
        true_box_grid, true_box_mask = create_true_box_grid(y)
        true_box_grids.append(true_box_grid)
        true_box_masks.append(true_box_mask)
    true_box_grids = tf.stack(true_box_grids)
    true_box_masks = tf.stack(true_box_masks)
    print('End data preprocessing')
    
    #dataset = dataset.padded_batch(params.batch_size, padded_shapes=([None, None, 3], [None, None]))

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
    for x, normalized_box_grid, orig_boxes in dataset:
        #print(x)
        #print(y)
        loss_value, grads = grad(model, x, normalized_box_grid)
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