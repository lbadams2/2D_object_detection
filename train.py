import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from detect_net import DetectNet, create_model
import params


# true box mask is 2-dim
def debug_output(true_box_grid, true_box_mask, pred_class_probs, pred_coords):
    zero = tf.zeros_like(true_box_grid)
    # where will be same shape as true_box_grid with true or false in each cell if not equal to zero
    where = tf.not_equal(true_box_grid, zero)            
    indices = tf.where(where)
    grid_indices = indices[:,:4]
    print('debug function - number of objects should be number of non zero values divided by 5 - number of objects in entire batch')
    print('number of non zero values true box grid using zero mask', grid_indices.shape)
    print('')

    expanded_mask = tf.tile(true_box_mask, [1,1,1,1,3])
    expanded_mask = expanded_mask[...,:5]
    masked_grid = true_box_grid * expanded_mask
    where = tf.not_equal(masked_grid, zero)
    indices = tf.where(where)
    grid_mask_indices = indices[:,:4]
    print('number of non zero values true box grid using true box mask', grid_mask_indices.shape)
    print('')

    test_probs = true_box_mask[..., :1] * pred_class_probs
    where = tf.not_equal(test_probs, zero)
    indices = tf.where(where)
    pred_prob_indices = indices[:,:4]
    print('number of non zero values pred class probs using true box mask', pred_prob_indices.shape)
    print('')

    print('number of objects for coords should be number of non zero values divided by 4, coord vec len 4, not 5 like previous 3')
    zero = tf.zeros_like(pred_coords)
    test_coords = true_box_mask[..., :1] * pred_coords
    where = tf.not_equal(test_coords, zero)
    indices = tf.where(where)
    pred_coord_indices = indices[:,:4]
    print('number of non zero values pred class probs using true box mask', pred_coord_indices.shape)
    print('')

    true_vecs = tf.gather_nd(true_box_grid, grid_indices)
    print('printing one true grid vector with object', grid_indices[0])
    print(true_vecs[0])
    print('')

    pred_class_vecs = tf.gather_nd(pred_class_probs, grid_indices)
    print('printing corresponding pred class probs vector', grid_indices[0])
    print(pred_class_vecs[0])
    print('')

    pred_coord_vecs = tf.gather_nd(pred_coords, grid_indices)
    print('printing corresponding pred coord vector', grid_indices[0])
    print(pred_coord_vecs[0])
    print('')
    
    uniques, idx, counts = tf.unique_with_counts(true_vecs[:,4])
    print('object types in batch', uniques)
    print('object counts by type', counts)
    print('')
    



# only penalize classification grid cell using SSE
# only penalize coordinate loss if object present in grid cell and box responsible for that object
# for cells with no object penalize classification score using SSE
# for boxes in grid cell that aren't responsible for object (only 1 if 2 anchor boxes) do SSE on object confidence score
# sum losses for all grid cells
def loss(model, x, true_box_grid, true_box_mask, training, count):
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

    # could use weight here on obj loss
    obj_conf_loss = params.obj_loss_weight * true_box_mask[...,:1] * tf.math.square(1 - obj_scores)
    conf_loss = no_obj_loss + obj_conf_loss

    matching_classes = tf.cast(true_box_grid[...,4], tf.int32)
    matching_classes = tf.one_hot(matching_classes, params.num_classes)
    class_loss = true_box_mask[..., :1] * tf.math.square(matching_classes - class_probs)

    # keras_yolo does a sigmoid on center_coords here but they should already be between 0 and 1 from predict_transform
    pred_boxes = tf.concat([center_coords, wh_coords], axis=-1)
    
    if count % 15 == 0:
        debug_output(true_box_grid, true_box_mask, class_probs, pred_boxes)
    
    matching_boxes = true_box_grid[..., :4]
    coord_loss = params.coord_loss_weight * true_box_mask[..., :1] * tf.math.square(matching_boxes - pred_boxes)

    confidence_loss_sum = tf.keras.backend.sum(conf_loss)
    classification_loss_sum = tf.keras.backend.sum(class_loss)
    coordinates_loss_sum = tf.keras.backend.sum(coord_loss)

    # not sure why .5 is here, maybe to make sure numbers don't get too large
    total_loss = 0.5 * (confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)            

    return total_loss


def grad(model, inputs, true_box_grid, box_mask, count):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, true_box_grid, box_mask, True, count)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def comp_nms_gt(nms_boxes, gt_boxes):
    pass


def _parse_image_function(example):
    # Create a dictionary describing the features.
    context_feature = {
        'image': tf.io.FixedLenFeature([], dtype=tf.string),
        'true_grid': tf.io.FixedLenFeature([4225], dtype=tf.float32),
        'mask_grid': tf.io.FixedLenFeature([1690], dtype=tf.float32)
    }
    sequence_features = {
        'Box Vectors': tf.io.VarLenFeature(dtype=tf.float32)
    }
    context_data, sequence_data = tf.io.parse_single_sequence_example(serialized=example, 
                                    context_features=context_feature, sequence_features=sequence_features)

    return context_data['image'], context_data['true_grid'], context_data['mask_grid'], sequence_data['Box Vectors']


def format_data(image, true_box_grid, mask_grid, labels):
    vecs = tf.sparse.to_dense(labels)
    #print(vecs)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # this should also normalize pixels
    image = tf.image.resize(image, [params.scaled_height, params.scaled_width])
    true_box_grid = tf.reshape(true_box_grid, [params.grid_height, params.grid_width, params.num_anchors, params.true_vec_len])
    mask_grid = tf.reshape(mask_grid, [params.grid_height, params.grid_width, params.num_anchors, 2])
    #print(image)
    return image, true_box_grid, mask_grid, vecs


def remove_orig_boxes(image, true_box_grid, mask_grid, labels):
    return image, true_box_grid, mask_grid


def get_dataset():
    FILENAME = 'image_grid_dataset_train.tfrecord'
    dataset = tf.data.TFRecordDataset(FILENAME)   
    dataset = dataset.map(_parse_image_function)
    dataset = dataset.map(format_data)

    dataset = dataset.map(remove_orig_boxes)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.shuffle(params.batch_size * 2)
    return dataset


# each object in training image is assigned to grid cell that contains object's midpoint
# and anchor box for the grid cell with highest IOU
def train_tf():
    dataset = get_dataset()
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
    count = 0
    for x, true_box_grid, box_mask in dataset:
        #print(x.shape, true_box_grid.shape, box_mask.shape)
        loss_value, grads = grad(model, x, true_box_grid, box_mask, count)
        print('train loss is ', loss_value)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        count += 1


def train_keras():
    dataset = get_dataset()
    model = create_model()
    model.compile(optimizer='adam',
              loss=loss)

    history = model.fit(dataset, epochs=10)


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
    train_keras()