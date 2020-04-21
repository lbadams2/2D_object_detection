from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization, LeakyReLU
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np
import params

'''
Image divided into dim/stride grid which corresponds to the output feature map
The cell in the output feature map that contains the center of the ground truth box is responsible for the object
Every image has 2 or 3 anchors (anchors are static), and each cell will create a bounding box for each anchor, so 2-3 boxes per cell
one to one mapping between bounding box within cell and anchor, anchor with highest IOU on ground truth box
will decide the bounding box used for the object
Network predicts the boudning box's offset to the anchor, not the dimensions of the bounding box
Each grid cell is normalized to have dim 1, part of the prediction will be (c_x, c_y) which is offset from top left of the grid cell,
(c_x, c_y) must be passed through a sigmoid to ensure they are less than 1 which forces them to stay in the cell,
these transformed center coordinates are used as the center of the bounding box
The network also outputs a width and height (t_w, t_h), these are passed to a log transform and then multiplied by the 
dimensions of the anchor used by the bounding box to get (b_w, b_h).
The center and width-height coordinates are then multiplied by feature map dims to get real coordinates

If stride of 32 is used on (1920 x 1280) image then grid is (60 x 40), 2 anchors means 2 bounding boxes per grid cell
for 60 x 40 x 2 = 4800 bounding boxes
In addition to coordinates, each box will also predict an objectness score and 3 confidence scores(for each class of object)
The objectness score for a box responsible for and object and boxes surrounding it will be close to 1, while farther boxes
should be closer to 0.
In order to filter 4800 boxes down to 1, throw out all boxes below some fixed objectness score threshold
'''


def create_model():
    model = models.Sequential()
    model.add(Conv2D(6, 3, padding='same', data_format='channels_last', kernel_regularizer=l2(5e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D())
    model.add(Conv2D(8, 3, padding='same', data_format='channels_last', kernel_regularizer=l2(5e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D())

    model.add(Conv2D(12, 3, padding='same', data_format='channels_last', kernel_regularizer=l2(5e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(8, 1, padding='same', data_format='channels_last', kernel_regularizer=l2(5e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(12, 3, padding='same', data_format='channels_last', kernel_regularizer=l2(5e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D())

    '''
    model.add(Conv2D(256, 3, padding='same', data_format='channels_last', kernel_regularizer=l2(5e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(128, 1, padding='same', data_format='channels_last', kernel_regularizer=l2(5e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(256, 3, padding='same', data_format='channels_last', kernel_regularizer=l2(5e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    '''

    model.add(Flatten())
    model.add(Dense(params.grid_height * params.grid_width * params.pred_vec_len * params.num_anchors, activation='relu'))

    return model    



# could resize image to make it square
class DetectNet():
    @staticmethod
    def get_new_dim(input_dim, kern_sz, pdng, strd):
        new_dim = int((input_dim - kern_sz + 2*pdng) / strd) + 1
        return new_dim
    

    # each anchor should be in the range grid_width x grid_height
    @staticmethod
    def get_anchors():
        #anchors = tf.convert_to_tensor(params.YOLO_ANCHORS, np.float32)
        anchors = tf.constant(params.YOLO_ANCHORS, dtype=tf.float32)
        return anchors


    @staticmethod
    def filter_boxes(transformed_predictions):
        center_coords, wh_coords, obj_scores, class_probs = transformed_predictions
        
        # get bounding boxes from center and wh coords
        box_mins = center_coords - (wh_coords / 2.)
        box_maxes = center_coords + (wh_coords / 2.)
        batch_boxes = K.concatenate([
            box_mins[..., 1:2],
            box_mins[..., 0:1],
            box_maxes[..., 1:2],
            box_maxes[..., 0:1]
        ])
        
        # print(batch_boxes.shape)

        # 0 out all the boxes that have confidence less than object_conf param
        box_scores = obj_scores * class_probs
        box_classes = K.argmax(box_scores, axis=-1)
        box_class_scores = K.max(box_scores, axis=-1)
        prediction_mask = box_class_scores >= params.object_conf

        # print('mask', box_scores.shape, box_classes.shape, box_class_scores.shape, prediction_mask.shape)

        image_dims = K.stack([params.scaled_height, params.scaled_width, params.scaled_height, params.scaled_width])
        image_dims = K.cast(K.reshape(image_dims, [1, 4]), dtype='float32')
        max_boxes_tensor = K.variable(params.max_boxes, dtype='int32')
        
        pred_boxes = []
        pred_scores = []
        pred_classes = []

        # loop through batch
        for mask, box, score, cls in zip(prediction_mask, batch_boxes, box_class_scores, box_classes):
            boxes = tf.boolean_mask(box, mask)
            scores = tf.boolean_mask(score, mask)
            classes = tf.boolean_mask(cls, mask)
        
            # print(boxes.shape, scores.shape, classes.shape)

            # scale bounding boxes to image size
            boxes = boxes * image_dims

            # non-max-suppression  
            nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=params.nms_conf)
            boxes = K.gather(boxes, nms_index)
            scores = K.gather(scores, nms_index)
            classes = K.gather(classes, nms_index)
            
            pred_boxes.append(boxes)
            pred_scores.append(scores)
            pred_classes.append(classes)
            
        return np.array(pred_boxes), np.array(pred_scores), np.array(pred_classes)



    # predictions will be (batch, grid_height, grid_width, num_anchors * vec_len)
    @staticmethod
    def predict_transform(predictions):
        predictions = tf.reshape(predictions, [-1, params.grid_height, params.grid_width, params.num_anchors, params.pred_vec_len])

        conv_dims = predictions.shape[1:3]
        conv_height_index = tf.keras.backend.arange(0, stop=conv_dims[0])
        conv_width_index = tf.keras.backend.arange(0, stop=conv_dims[1])
        conv_height_index = tf.tile(conv_height_index, [conv_dims[1]]) # (169,) tensor with 0-12 repeating
        conv_width_index = tf.tile(tf.expand_dims(conv_width_index, 0), [conv_dims[0], 1]) # (13, 13) tensor with x offset in each row
        conv_width_index = tf.keras.backend.flatten(tf.transpose(conv_width_index)) # (169,) tensor with 13 0's followed by 13 1's, etc (y offsets)
        conv_index = tf.transpose(tf.stack([conv_height_index, conv_width_index])) # (169, 2)
        conv_index = tf.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2]) # y offset, x offset
        conv_dims = tf.cast(tf.reshape(conv_dims, [1, 1, 1, 1, 2]), tf.float32) # grid_height x grid_width, max dims of anchors

        # makes the center coordinate between 0 and 1, each grid cell is normalized to 1 x 1
        center_coords = tf.math.sigmoid(predictions[...,:2])
        conv_index = tf.cast(conv_index, tf.float32)
        center_coords = (center_coords + conv_index) / conv_dims

        # makes the objectness score a probability between 0 and 1
        obj_scores = tf.math.sigmoid(predictions[...,4:5])
        
        anchors = DetectNet.get_anchors()
        anchors = tf.reshape(anchors, [1, 1, 1, params.num_anchors, 2])
        # exp to make width and height positive then multiply by anchor dims to resize box to anchor
        # should fit close to anchor, normalizing by conv_dims should make it between 0 and approx 1
        wh_coords = (tf.math.exp(predictions[...,2:4])*anchors) / conv_dims

        # apply sigmoid to class scores to make them probabilities
        class_probs = tf.keras.activations.softmax(predictions[..., 5 : 5 + params.num_classes])
        
        # (batch, rows, cols, anchors, vals)
        return center_coords, wh_coords, obj_scores, class_probs
