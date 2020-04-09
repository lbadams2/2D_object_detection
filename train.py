import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from detect_net import DetectNet
import params
from create_dataset import draw_orig_image


true_box_img_dict = {}

# only penalize classification grid cell using SSE
# only penalize coordinate loss if object present in grid cell and box responsible for that object
# for cells with no object penalize classification score using SSE
# for boxes in grid cell that aren't responsible for object (only 1 if 2 anchor boxes) do SSE on object confidence score
# sum losses for all grid cells
def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_, _ = model(x, training=training)
    # (batch, rows, cols, anchors, vals)
    center_coords, wh_coords, obj_scores, class_probs = DetectNet.predict_transform(y_)
    total_loss = 0

    center_coords_shape = center_coords.shape
    class_probs_shape = class_probs.shape
    true_grid_coords = (y[:,:,:2] // params.grid_stride) % 60
    
    # add remaining vals back
    true_grid_vals = tf.concat([true_grid_coords, y[:,:,2:]], axis=2)

    true_grid_coords = tf.cast(true_grid_coords, tf.int32)
    s1 = tf.shape(true_grid_coords, out_type=true_grid_coords.dtype)
    b = tf.range(s1[0])
    # Repeat batch index for each object
    b = tf.repeat(b, s1[1]) # (64) 16 times num_obj
    # Concatenate with row and column indices - (64, 1) concat (64, 2) = (64, 3), b is first col, true coords second, third col
    # vals in first col(each batch index 0-15) repeated 4 times from repeat function
    idx = tf.concat([tf.expand_dims(b, 1), tf.reshape(true_grid_coords, [-1, s1[2]])], axis=1)
    # Make mask by scattering values
    s2 = tf.shape(center_coords) # just need s2 to be (batch, rows, cols, depth, 2) for coord mask, last dim 5 for class mask
    # idx (64, 3), ones_like (64), s2[:3] (16, 40 60) = mask (16, 40, 60)
    # the vals of idx index into s2[:3], and set each to 1 (or whatever value is in middle arg)
    mask = tf.scatter_nd(idx, tf.ones_like(b, dtype=tf.float32), s2[:3])
    # Tile mask across last two dimensions
    s3 = tf.shape(class_probs)
    true_obj_coord_mask = tf.tile(mask[..., tf.newaxis, tf.newaxis], [1, 1, 1, s2[3], s2[4]])
    true_obj_class_mask = tf.tile(mask[..., tf.newaxis, tf.newaxis], [1, 1, 1, s3[3], s3[4]])

    true_obj_vals_batch = None
    for i in range(params.batch_size):
        img = x[i]
        true_obj_vals = true_box_img_dict[img.id]
        if true_obj_vals_batch is None:
            true_obj_vals_batch = true_obj_vals
        else:
            true_obj_vals_batch = tf.concat([true_obj_vals_batch, true_obj_vals], axis=0)


    pred_wh_half = wh_coords / 2.
    # bottom left corner
    pred_mins = center_coords - pred_wh_half
    # top right corner
    pred_maxes = center_coords + pred_wh_half

    true_xy = true_obj_vals[..., 0:2]
    true_wh = true_obj_vals[..., 2:4]
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
    no_obj_weights = params.noobj_loss_weight * (1 - object_detections) * (1 - true_obj_coord_mask)
    no_obj_loss = no_obj_weights * tf.math.square(obj_scores)

    obj_conf_loss = params.coord_loss_weight * true_obj_coord_mask * tf.math.square(1 - obj_scores)
    conf_loss = no_obj_loss + obj_conf_loss
    class_loss = true_obj_class_mask * tf.math.square(true_obj_vals[..., 5:] - class_probs)
    coord_loss = true_obj_coord_mask * tf.math.square(true_obj_vals[..., 0:2] - center_coords)
    coord_loss += true_obj_coord_mask * tf.math.square(true_obj_vals[..., 2:4] - wh_coords)

    confidence_loss_sum = tf.keras.backend.sum(conf_loss)
    classification_loss_sum = tf.keras.backend.sum(class_loss)
    coordinates_loss_sum = tf.keras.backend.sum(coord_loss)
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
    #print(image)
    return image, vecs


# tensors are (rows, cols)
# coords are (x, y) - opposite
def create_true_box_grid(image, labels):    
    anchors = DetectNet.create_anchors()
    for i, boxes in enumerate(labels):
        true_boxes = tf.constant(0, shape=[params.grid_height, params.grid_width, params.num_anchors, params.vec_len])
        true_boxes = tf.Variable(true_boxes)        

        boxes_coords = boxes[:,:4] * params.img_scale_factor
        boxes_coords_norm = boxes_coords / params.grid_stride # on same scale as anchors now
        boxes_grids = boxes_coords[:,:2] // params.grid_stride
        num_objs = boxes.shape[0].numpy()
        for obj in range(num_objs):
            best_iou = 0
            best_anchor = 0
            box_norm = boxes_coords_norm[obj]
            tensor_ind_y = boxes_grids[obj][0]
            tensor_ind_x = boxes_grids[obj][1]
            
            # find best anchor for current box
            for k, anchor in enumerate(anchors):
                # center box
                box_max = boxes_coords_norm[2:] / 2
                box_min = -box_max

                # center anchor
                anchor_max = anchor / 2
                anchor_min = -anchor_max

                intersect_mins = tf.math.maximum(box_min, anchor_min)
                intersect_maxes = tf.math.minimum(box_max, anchor_max)
                intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0)
                intersect_area = intersect_wh[0] * intersect_wh[1]
                box_area = boxes_coords_norm[2] * boxes_coords_norm[3]
                anchor_area = anchor[0] * anchor[1]
                iou = intersect_area / (box_area + anchor_area - intersect_area)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = k

            # its possible 2 objects in same cell have same anchor, last will overwrite
            if best_iou > 0:
                box_vec = tf.constant(0, shape=[params.vec_len])
                box_vec = tf.Variable(box_vec)
                box_vec[0].assign(box_norm[0] - tensor_ind_x) # center should be between 0 and 1, like prediction will be
                box_vec[1].assign(box_norm[1] - tensor_ind_y) # center should be between 0 and 1, like prediction will be
                adjusted_width = box_norm[2] / anchor[0]
                box_vec[2].assign(adjusted_width)
                adjusted_height = box_norm[3] / anchor[1]
                box_vec[3].assign(adjusted_height)
                true_boxes[i, tensor_ind_y, tensor_ind_x, k, :].assign(box_vec)
                
        true_box_img_dict[image.id] = true_boxes


def resize_images(image, labels):
    image = tf.image.resize(image, [416, 624])
    return image, labels
        

# each object in training image is assigned to grid cell that contains object's midpoint
# and anchor box for the grid cell with highest IOU
def train():
    FILENAME = 'image_dataset_train.tfrecord'
    dataset = tf.data.TFRecordDataset(FILENAME)    
    dataset = dataset.map(_parse_image_function)
    dataset = dataset.map(format_data)
    
    print('Begin data preprocessing')
    dataset = dataset.map(resize_images)
    dataset = dataset.map(create_true_box_grid)
    print('End data preprocessing')
    
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