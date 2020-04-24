import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from detect_net import DetectNet, create_model, create_darknet_model
import params
from tqdm import tqdm

# true box mask is 2-dim


def debug_output(true_box_grid, pred_class_probs, pred_coords, true_box_mask=None, mem_mask=None):
    zero = tf.zeros_like(true_box_grid)
    # where will be same shape as true_box_grid with true or false in each cell if not equal to zero
    where = tf.not_equal(true_box_grid, zero)
    indices = tf.where(where)
    grid_indices = indices[:, :4]
    print('debug function - number of objects should be number of non zero values divided by 5 - number of objects in entire batch')
    print('number of non zero values true box grid using zero mask',
          grid_indices.shape)
    print('')

    if true_box_mask is not None:
        expanded_mask = tf.tile(true_box_mask, [1, 1, 1, 1, 3])
        expanded_mask = expanded_mask[..., :5]
        masked_grid = true_box_grid * expanded_mask
        where = tf.not_equal(masked_grid, zero)
        indices = tf.where(where)
        grid_mask_indices = indices[:, :4]
        print('number of non zero values true box grid using true box mask',
              grid_mask_indices.shape)
        print('')

        test_probs = true_box_mask[..., :1] * pred_class_probs
        test_coords = true_box_mask[..., :1] * pred_coords

    if mem_mask is not None:
        mem_masked_grid = true_box_grid * mem_mask
        where = tf.not_equal(mem_masked_grid, zero)
        indices = tf.where(where)
        mem_mask_indices = indices[:, :4]
        print('number of non zero values true box grid using mem mask',
              grid_mask_indices.shape)
        print('')

        test_probs = mem_mask[..., :1] * pred_class_probs
        test_coords = mem_mask[..., :1] * pred_coords

    #test_probs = true_box_mask[..., :1] * pred_class_probs
    where = tf.not_equal(test_probs, zero)
    indices = tf.where(where)
    pred_prob_indices = indices[:, :4]
    print('number of non zero values pred class probs using true box mask',
          pred_prob_indices.shape)
    print('')

    print('number of objects for coords should be number of non zero values divided by 4, coord vec len 4, not 5 like previous 3')
    zero = tf.zeros_like(pred_coords)
    #test_coords = true_box_mask[..., :1] * pred_coords
    where = tf.not_equal(test_coords, zero)
    indices = tf.where(where)
    pred_coord_indices = indices[:, :4]
    print('number of non zero values pred class probs using true box mask',
          pred_coord_indices.shape)
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

    uniques, idx, counts = tf.unique_with_counts(true_vecs[:, 4])
    print('object types in batch', uniques)
    print('object counts by type', counts)
    print('')


def create_mask(true_box_grid):
    zero = tf.zeros_like(true_box_grid)
    where = tf.not_equal(true_box_grid, zero)
    mask = tf.cast(where, tf.float32)
    return mask


# only penalize classification grid cell using SSE
# only penalize coordinate loss if object present in grid cell and box responsible for that object
# for cells with no object penalize classification score using SSE
# for boxes in grid cell that aren't responsible for object (only 1 if 2 anchor boxes) do SSE on object confidence score
# sum losses for all grid cells
def loss_custom(x, true_box_grid, model=None, true_box_mask=None, training=True, count=-1):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    # (batch, rows, cols, anchors, vals)
    center_coords, wh_coords, obj_scores, class_probs = DetectNet.predict_transform(
        y_)
    detector_mask = create_mask(true_box_grid)
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

    print('max prediction/true iou', np.amax(best_ious.numpy()))

    # A detector has found an object if IOU > thresh for some true box.
    object_detections = tf.keras.backend.cast(
        best_ious > params.iou_thresh, dtype=tf.float32)

    # for both no obj and obj only calculate loss for boxes that had high ious
    no_obj_weights = params.noobj_loss_weight * \
        (1 - object_detections) * (1 - detector_mask[..., :1])
    no_obj_loss = no_obj_weights * tf.math.square(obj_scores)

    # could use weight here on obj loss
    obj_conf_loss = params.obj_loss_weight * \
        detector_mask[..., :1] * tf.math.square(1 - obj_scores)
    conf_loss = no_obj_loss + obj_conf_loss

    matching_classes = tf.cast(true_box_grid[..., 4], tf.int32)
    matching_classes = tf.one_hot(matching_classes, params.num_classes)
    class_loss = detector_mask[..., :1] * \
        tf.math.square(matching_classes - class_probs)

    # keras_yolo does a sigmoid on center_coords here but they should already be between 0 and 1 from predict_transform
    pred_boxes = tf.concat([center_coords, wh_coords], axis=-1)

    # if count % 15 == 0:
    #    debug_output(true_box_grid, class_probs, pred_boxes,
    #                 true_box_mask, detector_mask)

    matching_boxes = true_box_grid[..., :4]
    coord_loss = params.coord_loss_weight * \
        detector_mask[..., :1] * tf.math.square(matching_boxes - pred_boxes)

    confidence_loss_sum = tf.keras.backend.sum(conf_loss)
    classification_loss_sum = tf.keras.backend.sum(class_loss)
    coordinates_loss_sum = tf.keras.backend.sum(coord_loss)

    # not sure why .5 is here, maybe to make sure numbers don't get too large
    total_loss = 0.5 * (confidence_loss_sum +
                        classification_loss_sum + coordinates_loss_sum)

    return total_loss


def loss_keras(y_, true_box_grid):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    #y_ = model(x, training=training)
    # (batch, rows, cols, anchors, vals)
    center_coords, wh_coords, obj_scores, class_probs = DetectNet.predict_transform(
        y_)
    detector_mask = create_mask(true_box_grid)
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
    object_detections = tf.keras.backend.cast(
        best_ious > 0.6, dtype=tf.float32)

    # for both no obj and obj only calculate loss for boxes that had high ious
    no_obj_weights = params.noobj_loss_weight * \
        (1 - object_detections) * (1 - detector_mask[..., :1])
    no_obj_loss = no_obj_weights * tf.math.square(obj_scores)

    # could use weight here on obj loss
    obj_conf_loss = params.obj_loss_weight * \
        detector_mask[..., :1] * tf.math.square(1 - obj_scores)
    conf_loss = no_obj_loss + obj_conf_loss

    matching_classes = tf.cast(true_box_grid[..., 4], tf.int32)
    matching_classes = tf.one_hot(matching_classes, params.num_classes)
    class_loss = detector_mask[..., :1] * \
        tf.math.square(matching_classes - class_probs)

    # keras_yolo does a sigmoid on center_coords here but they should already be between 0 and 1 from predict_transform
    pred_boxes = tf.concat([center_coords, wh_coords], axis=-1)

    #debug_output(true_box_grid, class_probs, pred_boxes, None, detector_mask)

    matching_boxes = true_box_grid[..., :4]
    coord_loss = params.coord_loss_weight * \
        detector_mask[..., :1] * tf.math.square(matching_boxes - pred_boxes)

    confidence_loss_sum = tf.keras.backend.sum(conf_loss)
    classification_loss_sum = tf.keras.backend.sum(class_loss)
    coordinates_loss_sum = tf.keras.backend.sum(coord_loss)

    # not sure why .5 is here, maybe to make sure numbers don't get too large
    total_loss = 0.5 * (confidence_loss_sum +
                        classification_loss_sum + coordinates_loss_sum)

    return total_loss


def grad(model, inputs, true_box_grid, box_mask, count):
    with tf.GradientTape() as tape:
        loss_value = loss_custom(
            inputs, true_box_grid, model, box_mask, True, count)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


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
    # print(vecs)
    image = tf.image.decode_jpeg(image)
    # this should also normalize pixels
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [params.scaled_height, params.scaled_width])
    true_box_grid = tf.reshape(true_box_grid, [
                               params.grid_height, params.grid_width, params.num_anchors, params.true_vec_len])
    mask_grid = tf.reshape(
        mask_grid, [params.grid_height, params.grid_width, params.num_anchors, 2])
    # print(image)
    return image, true_box_grid, mask_grid, vecs


def remove_orig_boxes(image, true_box_grid, mask_grid, labels):
    return image, true_box_grid, mask_grid


def remove_mask(image, true_box_grid, mask_grid):
    return image, true_box_grid


def get_dataset(file_name):
    dataset = tf.data.TFRecordDataset(file_name)
    dataset = dataset.map(_parse_image_function)
    dataset = dataset.map(format_data)

    dataset = dataset.map(remove_orig_boxes)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.shuffle(params.batch_size * 2)
    return dataset


def iou(pred_boxes, true_boxes):
    # max bottom left corner
    intersect_mins = np.maximum(pred_boxes[:,:2], true_boxes[:,:2])
    # min top right corner
    intersect_maxes = np.minimum(pred_boxes[:,2:4], true_boxes[:,2:4])
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    # product of difference between x max and x min, y max and y min
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_areas = (pred_boxes[:,2] - pred_boxes[:,0]) * (pred_boxes[:,3] - pred_boxes[:,1])
    true_areas = (true_boxes[:,2] - true_boxes[:,0]) * (true_boxes[:,3] - true_boxes[:,1])

    union_areas = pred_areas + true_areas - intersect_areas
    ious = intersect_areas / union_areas

    return ious


def get_metrics(pred_boxes, pred_scores, pred_classes, pred_grid_indexes, true_grid):
    mask = create_mask(true_grid)    
    masked_grid = true_grid * mask
    num_true_positives = 0
    for i, img_boxes in enumerate(pred_boxes):        
        img_true_grid = masked_grid[i]
        zero = tf.zeros_like(img_true_grid)
        where = tf.not_equal(img_true_grid, zero)
        indices = tf.where(where)
        grid_indices = indices[:, :3]
        uniques = np.unique(grid_indices.numpy(), axis=0)
        img_num_true_boxes = uniques.shape[0]
        img_true_grid = img_true_grid.numpy()

        img_pred_boxes = pred_boxes[i].numpy()
        img_pred_scores = pred_scores[i].numpy()
        img_pred_classes = pred_classes[i].numpy()
        img_pred_inds = pred_grid_indexes[i].numpy()
        
        # get true positives
        pred_set = set([tuple(ind) for ind in img_pred_inds])
        true_set = set([tuple(ind) for ind in uniques])
        true_positive_inds = np.array([ind for ind in pred_set & true_set])
        num_tp = true_positive_inds.shape[0]

        tp_pred_boxes = np.array((num_tp, 5))
        tp_true_boxes = np.array((num_tp, 5))
        for j, tp_ind in enumerate(true_positive_inds):
            arr_ind = np.where(np.all(img_pred_inds == tp_ind, axis=1))
            pred_box = img_pred_boxes[arr_ind]
            pred_class_probs = img_pred_classes[arr_ind]
            class_index = np.argmax(pred_class_probs)
            pred_box = np.concatenate(pred_box, class_index)
            true_box = img_true_grid[tp_ind[0], tp_ind[1], tp_ind[2]]
            tp_pred_boxes[j] = pred_box
            tp_true_boxes[j] = true_box
        
        tp_class = tp_pred_boxes == tp_true_boxes
        ious = iou(tp_pred_boxes, tp_true_boxes)
        true_positives = ious[tp_class]
        true_positives = np.where(true_positives > params.iou_thresh)
        num_true_positives += true_positives.size



def run_validation(val_dataset, model):
    print('running validation set')
    for val_x, val_grid, val_mask in val_dataset:
        logits = model(val_x, training=False)
        transformed_pred = DetectNet.predict_transform(logits)
        pred_boxes, pred_scores, pred_classes, pred_grid_indexes = DetectNet.filter_boxes(
            transformed_pred)
        if np.array(pred_boxes).size > 0:
            get_metrics(pred_boxes, pred_scores, pred_classes,
                        pred_grid_indexes, val_grid)


# each object in training image is assigned to grid cell that contains object's midpoint
# and anchor box for the grid cell with highest IOU
def train(train_dataset, val_dataset, model, epochs=1):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    for e in tqdm(range(epochs)):
        # print('Epoch:{}/{}'.format(e+1, epochs))
        count = 0
        for x, true_box_grid, box_mask in train_dataset:
            #print(x.shape, true_box_grid.shape, box_mask.shape)
            loss_value, grads = grad(model, x, true_box_grid, box_mask, count)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print('Loss:', loss_value)

            # run on cpu?
            if count % 10 == 0:
                run_validation(val_dataset, model)

            count += 1

    return model


def main():
    train_dataset = get_dataset('image_grid_dataset_train.tfrecord')
    val_dataset = get_dataset('validation.tfrecord')
    model = create_darknet_model()
    # print(model.summary())
    model = train(train_dataset, val_dataset, model, epochs=params.epochs)


def train_keras():
    dataset = get_dataset()
    dataset = dataset.map(remove_mask)
    model = create_model()
    model.compile(optimizer='adam',
                  loss=loss_keras)

    history = model.fit(dataset, epochs=10)


def print_results(model, dataset):
    history = model.fit_generator(
        dataset,
        steps_per_epoch=total_train,  # batch_size,
        epochs=params.epochs,
        validation_data=val_dataset,
        validation_steps=total_val  # batch_size
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

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
    # train_keras()
    main()
