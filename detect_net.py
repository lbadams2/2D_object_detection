from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras import layers
import tensorflow.keras.backend as K
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


# could resize image to make it square
class DetectNet(layers.Layer):
    def __init__(self, training):
        super(DetectNet, self).__init__()
        self.training = training

        # conv2D args are filters, kernel_size, ...
        # filters is number of output channels
        # channels_last means input is (batch, height, width, channels)
        self.conv_1 = Conv2D(32, params.grid_stride, strides=2, padding='valid', data_format='channels_last', activation='relu')
        self.pool_1 = MaxPool2D(pool_size=params.pool_size)
        self.conv_2 = Conv2D(64, params.kernel_size, strides=params.kernel_size, padding='valid', data_format='channels_last', activation='relu')
        self.pool_2 = MaxPool2D(pool_size=params.pool_size)
        self.conv_3 = Conv2D(128, params.kernel_size, strides=params.kernel_size, padding='valid', data_format='channels_last', activation='relu')
        self.flatten_layer = Flatten()
        self.dropout = Dropout(0.4)
        self.linear_1 = Dense(params.grid_height * params.grid_width * params.pred_vec_len * params.num_anchors, activation='relu')
        #self.linear_2 = Dense(params.vec_len) # 4 coordinates, objectness score, 3 class probs
        # output here should be (batch_sz, height/grid_stride, width/grid_stride, num_anchors * vec_len



    # img should be np image array (1920 x 1280 x 3)
    def call(self, img):
        x = self.conv_1(img) # x is (16, 625, 945, 32) get_new_dim correct
        #x = tf.nn.relu(x)
        x = self.pool_1(x) # x is (16, 312, 472, 32) get_new_dim correct
        x = self.conv_2(x) # x is (16, 104, 157, 64) get_new_dim correct
        #x = tf.nn.relu(x)
        x = self.pool_2(x) # x is (16, 52, 78, 64) correct
        x = self.conv_3(x) # x is (16, 17, 26, 128)
        #x = tf.nn.relu(x)        
        x = self.flatten_layer(x) # x is (16, 56576) here
        x = self.dropout(x)
        x = self.linear_1(x)
        #x = self.dropout(x)
        #x = self.linear_2(x)

        # x = DetectNet.predict_transform(x)
        # now it should have anchors for (60 x 40 x 16)

        # this will filter 4800 boxes down to 1 box per object, output likely (3 x 7) or something
        # 8 to 7 b/c removes class probs for non predicted objects and adds class index for predicted object
        filtered_output = None
        if not self.training:
            x = DetectNet.predict_transform(x)
            filtered_output = DetectNet.filter_boxes(x)
        return x, filtered_output


    @staticmethod
    def get_new_dim(input_dim, kern_sz, pdng, strd):
        new_dim = int((input_dim - kern_sz + 2*pdng) / strd) + 1
        return new_dim
    

    # each anchor should be in the range grid_width x grid_height
    @staticmethod
    def get_anchors():
        #anchors = tf.convert_to_tensor(params.YOLO_ANCHORS, np.float32)
        anchors = tf.Variable(params.YOLO_ANCHORS, dtype=tf.float32)
        return anchors


    @staticmethod
    def filter_boxes(transformed_predictions):
        center_coords, wh_coords, obj_scores, class_probs = transformed_predictions
        
        # get bounding boxes from center and wh coords
        box_mins = center_coords - (wh_coords / 2.)
        box_maxes = center_coords + (wh_coords / 2.)
        boxes = K.concatenate([
            box_mins[..., 1:2],
            box_mins[..., 0:1],
            box_maxes[..., 1:2],
            box_maxes[..., 0:1]
        ])
        
        # 0 out all the boxes that have confidence less than object_conf param
        box_scores = obj_scores * class_probs
        box_classes = K.argmax(box_scores, axis=-1)
        box_class_scores = K.max(box_scores, axis=-1)
        prediction_mask = box_class_scores >= params.object_conf

        boxes = tf.boolean_mask(boxes, prediction_mask)
        scores = tf.boolean_mask(box_class_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)
        
        # scale bounding boxes to image size
        image_dims = K.stack([params.im_height, params.im_width, params.im_height, params.im_width])
        image_dims = K.reshape(image_dims, [1, 4])
        boxes = boxes * image_dims

        # non-max-suppression
        max_boxes_tensor = K.variable(params.max_boxes, dtype='int32')
        nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=params.nms_conf)
        boxes = K.gather(boxes, nms_index)
        scores = K.gather(scores, nms_index)
        classes = K.gather(classes, nms_index)
        
        return boxes, scores, classes


    # predictions will be (batch, grid_height, grid_width, num_anchors * vec_len)
    @staticmethod
    def predict_transform(predictions):
        predictions = tf.reshape(predictions, [params.batch_size, params.grid_height, params.grid_width, params.num_anchors, params.pred_vec_len])

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
        class_probs = tf.math.sigmoid(predictions[..., 5 : 5 + params.num_classes])
        
        # (batch, rows, cols, anchors, vals)
        return center_coords, wh_coords, obj_scores, class_probs