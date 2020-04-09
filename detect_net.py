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
        self.linear_1 = Dense(40 * 60 * params.vec_len * 2, activation='relu')
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
        x = tf.reshape(x, [params.batch_size, 40, 60, params.vec_len * 2])
        # x should be (batch_sz, 60 x 40 x 16) here

        #x = DetectNet.predict_transform(x, self.anchors_grid)
        # now it should have anchors for (60 x 40 x 16)

        # this will filter 4800 boxes down to 1 box per object, output likely (3 x 7) or something
        # 8 to 7 b/c removes class probs for non predicted objects and adds class index for predicted object
        filtered_output = None
        if not self.training:
            transformed_predictions = DetectNet.predict_transform(x)
            filtered_output = DetectNet.filter_boxes(transformed_predictions)
        return x, filtered_output


    @staticmethod
    def get_new_dim(input_dim, kern_sz, pdng, strd):
        new_dim = int((input_dim - kern_sz + 2*pdng) / strd) + 1
        return new_dim
    
    # each anchor should be in the range grid_width x grid_height (17 x 13)
    # multiply by grid_stride to coords on image
    @staticmethod
    def create_anchors():
        wide_anchor = tf.constant(0, dtype=tf.float32, shape=(2))
        wide_anchor = tf.Variable(wide_anchor)
        wide_anchor[0].assign(1) # width
        wide_anchor[1].assign(.5) # height

        tall_anchor = tf.constant(0, dtype=tf.float32, shape=(2))
        tall_anchor = tf.Variable(tall_anchor)
        tall_anchor[0].assign(.5) # width
        tall_anchor[1].assign(1) # height

        anchors = tf.stack([wide_anchor, tall_anchor])
        anchors = tf.transpose(anchors, perm=[1, 0])

        return anchors


    @staticmethod
    def bbox_iou(box, box_arr):
        box_corner = tf.Variable(box)
        # get bottom left corner
        box_corner[0].assign(box[0] - box[2]/2) # x center minus width/2
        box_corner[1].assign(box[1] - box[3]/2) # y center minus height/2
        # get top right corner
        box_corner[2].assign(box[0] + box[2]/2)
        box_corner[3].assign(box[1] + box[3]/2)

        box_arr = tf.Variable(box_arr)
        box_arr[:,0].assign(box_arr[:,0] - box_arr[:,2]/2) # x center minus width/2
        box_arr[:,1].assign(box_arr[:,1] - box_arr[:,3]/2) # y center minus height/2
        box_arr[:,2].assign(box_arr[:,0] + box_arr[:,2]/2)
        box_arr[:,3].assign(box_arr[:,1] + box_arr[:,3]/2)

        #Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box[0], box[1], box[2], box[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box_arr[:,0], box_arr[:,1], box_arr[:,2], box_arr[:,3]
        
        #get the corrdinates of the intersection rectangle
        # get max for bottom left corner (x1 and y1 were subtracted from)
        inter_rect_x1 =  tf.math.maximum(b1_x1, b2_x1)
        inter_rect_y1 =  tf.math.maximum(b1_y1, b2_y1)
        # get min for top right corner (x2 and y2 were added to)
        inter_rect_x2 =  tf.math.minimum(b1_x2, b2_x2)
        inter_rect_y2 =  tf.math.minimum(b1_y2, b2_y2)
        
        #Intersection area
        # the x and y offsets are defined to be 1 between grid cells in predict transform
        # 1 should be max difference, not sure why 1 is added, clip by value makes sure no negative numbers
        inter_area = tf.clip_by_value(inter_rect_x2 - inter_rect_x1 + 1, clip_value_min=0, clip_value_max=2000) * \
                                                    tf.clip_by_value(inter_rect_y2 - inter_rect_y1 + 1, clip_value_min=0, clip_value_max=2000)
    
        #Union Area
        b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
        
        iou = inter_area / (b1_area + b2_area - inter_area)
        
        return iou

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


    # this applies sigmoids and exponential function to some dims to make them between 0 and 1 or positive
    # also scales bounding boxes by anchor box dims
    @staticmethod
    def predict_transform(predictions):
        anchors = DetectNet.create_anchors()
        
        grid_size_x = predictions.shape[2]
        grid_size_y = predictions.shape[1]

        # makes the x_center coordinate between 0 and 1, each grid cell is normalized to 1 x 1
        b1_center_coords = tf.math.sigmoid(predictions[:,:,:,:2])
        b2_center_coords = tf.math.sigmoid(predictions[:,:,:,10:12])
        # makes the y_center coordinate between 0 and 1, each grid cell is normalized to 1 x 1
        #y_centers = tf.math.sigmoid(predictions[:,:,:,1])
        # makes the objectness score a probability between 0 and 1
        b1_obj_scores = tf.math.sigmoid(predictions[:,:,:,4])
        b2_obj_scores = tf.math.sigmoid(predictions[:,:,:,14])

        # np.arrange creates an array of ints from 0 to grid_size - 1
        grid_x = np.arange(grid_size_x)
        grid_y = np.arange(grid_size_y)
        # a contains 40 rows of x-offsets, each row identical counting up to 60
        # b contains 40 rows of y-offsets, first row 60 0's, second 60 1's, and so on up to 60 40's
        a,b = np.meshgrid(grid_x, grid_y)
        x_offset = tf.constant(a, dtype=tf.float32)
        y_offset = tf.constant(b, dtype=tf.float32)

        # concatenates the grids horizontally
        x_y_offset = tf.stack([x_offset, y_offset])
        x_y_offset = tf.transpose(x_y_offset, perm=[1,2,0])

        b1_center_coords = b1_center_coords + x_y_offset # center of first box
        b2_center_coords = b2_center_coords + x_y_offset

        anchors = tf.tile(anchors, [40, 60])
        anchors = tf.reshape(anchors, [40, 60, 4]) # make 2 anchors per bounding box
        # exp to make width and height positive then multiply by anchor dims to resize box to anchor
        b1_wh_coords = tf.math.exp(predictions[:,:,:,2:4])*anchors[:,:,:2]
        b2_wh_coords = tf.math.exp(predictions[:,:,:,12:14])*anchors[:,:,2:4]

        # apply sigmoid to class scores to make them probabilities
        # index 4 and 12 is the object conf score, this isn't being modified here
        b1_class_probs = tf.math.sigmoid(predictions[:,:,:, 5 : 5 + params.num_classes])
        b2_class_probs = tf.math.sigmoid(predictions[:,:,:, 15 : 15 + params.num_classes])

        # resize coordinates to size of input image
        #b1_center_coords = b1_center_coords * params.grid_stride
        #b2_center_coords = b2_center_coords * params.grid_stride
        #b1_wh_coords = b1_wh_coords * params.grid_stride
        #b2_wh_coords = b2_wh_coords * params.grid_stride

        center_coords = tf.stack([b1_center_coords, b2_center_coords])
        center_coords_shape = center_coords.shape
        center_coords = tf.reshape(center_coords, [center_coords_shape[1], center_coords_shape[2], 
                                                        center_coords_shape[3], center_coords_shape[0], center_coords_shape[4]])
        
        wh_coords = tf.stack([b1_wh_coords, b2_wh_coords])
        wh_coords_shape = wh_coords.shape
        wh_coords = tf.reshape(wh_coords, [wh_coords_shape[1], wh_coords_shape[2], 
                                                        wh_coords_shape[3], wh_coords_shape[0], wh_coords_shape[4]])

        obj_scores = tf.stack([b1_obj_scores, b2_obj_scores])
        obj_scores = tf.expand_dims(obj_scores, 4)
        obj_scores_shape = obj_scores.shape
        obj_scores = tf.reshape(obj_scores, [obj_scores_shape[1], obj_scores_shape[2], 
                                                        obj_scores_shape[3], obj_scores_shape[0], obj_scores_shape[4]])

        class_probs = tf.stack([b1_class_probs, b2_class_probs])
        class_probs_shape = class_probs.shape
        class_probs = tf.reshape(class_probs, [class_probs_shape[1], class_probs_shape[2], 
                                                        class_probs_shape[3], class_probs_shape[0], class_probs_shape[4]])
        # (batch, rows, cols, anchors, vals)
        return center_coords, wh_coords, obj_scores, class_probs