from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras import layers
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
        self.linear_1 = Dense(40 * 60 * params.vec_len * params.num_anchors, activation='relu')
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
        pred_idxs = None
        if not self.training:
            pred_idxs = DetectNet.filter_boxes(x)
        return x, pred_idxs


    @staticmethod
    def get_new_dim(input_dim, kern_sz, pdng, strd):
        new_dim = int((input_dim - kern_sz + 2*pdng) / strd) + 1
        return new_dim
    

    # each anchor should be in the range grid_width x grid_height
    @staticmethod
    def create_anchors():
        #anchors = tf.convert_to_tensor(params.YOLO_ANCHORS, np.float32)
        anchors = tf.Variable(params.YOLO_ANCHORS, dtype=tf.float32)
        return anchors


    # predictions will be (4800 x 8)
    @staticmethod
    def filter_boxes(predictions):
        predictions = tf.reshape(predictions, [60 * 40 * 2, params.vec_len])
        predictions = tf.Variable(predictions)
        conf_mask = tf.Variable(predictions[:,4] > params.object_conf)
        conf_mask = tf.dtypes.cast(conf_mask, tf.double)
        conf_mask = tf.ones([4800, params.vec_len], dtype=tf.double) * tf.expand_dims(conf_mask,1)
        predictions.assign(predictions * conf_mask) # this should 0 out all 8 box attrs if 4th elem below object threshold

        max_ind = tf.math.argmax(predictions[:,5:5+ params.num_classes], axis=1) # get index of max class prob for each box
        max_ind = tf.dtypes.cast(max_ind, tf.float64)
        max_ind = tf.expand_dims(max_ind, 1)
        max_score = tf.reduce_max(predictions[:,5:5+ params.num_classes], axis=1) # get value of max class prob for each box
        max_score = tf.dtypes.cast(max_score, tf.float64)
        max_score = tf.expand_dims(max_score, 1)
        seq = (predictions[:,:5], max_ind, max_score) # (4800 x 5, 1, 1)
        image_pred = tf.concat(seq, 1)
        # image_pred is 4800 x 7 containing coords and highest prob class and index
        
        zeros = tf.zeros([4800, 1], tf.float64)
        where = tf.not_equal(image_pred[:,0], zeros[:,0])
        nonzero_indices = tf.where(where)
        image_pred_ = tf.gather(image_pred, nonzero_indices, axis=0)
        image_pred_ = tf.squeeze(image_pred_, 1) # image_pred now only contains with object score greater than object_conf
        if image_pred_.shape[0] == 0:
            return # if there are no nonzero indices, no predictions for this image
        
        img_classes, _ = tf.unique(image_pred_[:,-2])
        output = None
        for cls_ind in img_classes:
            cls_mask = image_pred_[:,-2] == cls_ind
            cls_mask = tf.dtypes.cast(cls_mask, tf.double)
            cls_mask = tf.ones(image_pred_.shape, dtype=tf.double) * tf.expand_dims(cls_mask,1)
            cls_mask = image_pred_ * cls_mask # get boxes with prediction for this class
            class_mask_ind = tf.not_equal(cls_mask[:,-1], zeros[:image_pred_.shape[0],0])
            nonzero_indices = tf.where(class_mask_ind)
            image_pred_class = tf.gather(image_pred_, nonzero_indices, axis=0)
            if len(image_pred_class.shape) > 2:
                image_pred_class = tf.squeeze(image_pred_class, 1) # should be boxes that have a prediction for cls_ind

            _, sorted_inds = tf.math.top_k(image_pred_class[:,4], k=tf.size(image_pred_class[:,4]))
            image_pred_class = tf.gather(image_pred_class, sorted_inds, axis=0) # should be sorted by obj score descending now
            if len(image_pred_class.shape) > 2:
                image_pred_class = tf.squeeze(image_pred_class, 1)
            idx = image_pred_class.shape[0]

            # NMS - multiple grid cells may detect same object. Iterate from highest prob box to smallest and find IOU
            # between that box and all with lower confidence, for the boxes with IOU higher than threshold, remove
            # them from the tensor.  Tensor gets smaller each iteration but rows between 0 and i that weren't previously
            # removed will remain.
            for i in range(idx):
                try:
                    # get iou of ith box with all boxes after it
                    ious = DetectNet.bbox_iou(tf.expand_dims(image_pred_class[i], 1), image_pred_class[i+1:])
                # rows are being removed each iteration, eventually i will exceed size of tensor
                except ValueError as e:
                    break

                except (IndexError, tf.errors.InvalidArgumentError):
                    break
                
                iou_mask = ious < params.nms_conf
                iou_mask = tf.dtypes.cast(iou_mask, tf.double)
                iou_mask = tf.ones(image_pred_class[i+1:].shape, dtype=tf.double) * tf.expand_dims(iou_mask,1)
                iou_mask = image_pred_class[i+1:] * iou_mask # zero out 7 dim vectors for boxes less than threshold
                image_pred_class = tf.concat([image_pred_class[:i+1], iou_mask], 0) # put preceding i rows back in tensor
                # 0 to i will be non zero, i+1 and up are affected by the mask
                iou_mask_ind = tf.not_equal(image_pred_class[:,-1], zeros[:image_pred_class.shape[0], 0])
                nonzero_indices = tf.where(iou_mask_ind)
                # every iteration removes rows that overlap too much with a higher confidence prediction
                # nonzero_indices includes rows kept from previous iterations (0 to i)
                image_pred_class = tf.gather(image_pred_class, nonzero_indices, axis=0)
                if len(image_pred_class.shape) > 2:
                    image_pred_class = tf.squeeze(image_pred_class, 1)

            if output is None:
                output = image_pred_class
            else:
                output = tf.concat([output, image_pred_class], 0)
        
        return output


    # predictions will be (batch, grid_height, grid_width, num_anchors * vec_len)
    @staticmethod
    def predict_transform(predictions):
        predictions = tf.reshape(predictions, [params.batch_size, params.grid_height, params.grid_width, params.num_anchors, params.vec_len])
    
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
        obj_scores = tf.math.sigmoid(predictions[...,4])
        
        anchors = DetectNet.create_anchors()
        anchors = tf.reshape(anchors, [1, 1, 1, params.num_anchors, 2])
        # exp to make width and height positive then multiply by anchor dims to resize box to anchor
        # should fit close to anchor, normalizing by conv_dims should make it between 0 and approx 1
        wh_coords = (tf.math.exp(predictions[...,2:4])*anchors) / conv_dims

        # apply sigmoid to class scores to make them probabilities
        class_probs = tf.math.sigmoid(predictions[..., 5 : 5 + params.num_classes])
        
        # (batch, rows, cols, anchors, vals)
        return center_coords, wh_coords, obj_scores, class_probs