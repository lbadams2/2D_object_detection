from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
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
for 60 x 40 x 3 = 7200 bounding boxes
In addition to coordinates, each box will also predict an objectness score and 3 confidence scores(for each class of object)
The objectness score for a box responsible for and object and boxes surrounding it will be close to 1, while farther boxes
should be closer to 0.
In order to filter 4800 boxes down to 1, throw out all boxes below some fixed objectness score threshold

Use 3 anchor boxes, 1 bottom half of the image, 1 left half, and 1 right half
'''



# each grid cell has 3 boxes associated with it for (3 x (5 + num_classes)) entires in feature map
# 5 is 4 coordinates and objectness score
class DetectNet(layers.Layer):
    def __init__(self, width, height):
        super(DetectNet, self).__init__()
        self.width = width
        self.height = height
        self.grid_width = width / params.stride
        self.grid_height = height / params.stride

        self.conv_1 = Conv2D(32, 6, 6, input_shape=X.shape[1:], dim_ordering='tf', activation='relu')
        self.pool_1 = MaxPooling2D(pool_size=(params.pool_size, params.pool_size))
        self.conv_2 = Conv2D(64, params.filter_size, params.filter_size, dim_ordering='tf', activation='relu')
        self.pool_2 = MaxPooling2D(pool_size=(params.pool_size, params.pool_size))
        self.conv_3 = Conv2D(128, params.filter_size, params.filter_size, dim_ordering='tf', activation='relu')
        self.dropout = Dropout(0.4)
        self.linear_1 = Dense(256, activation='relu')
        self.linear_2 = Dense(8) # 4 coordinates, objectness score, 3 class probs

        self.anchors = self.create_anchors()

    # img should be np image array (1920 x 1280 x 3)
    def call(self, img):
        x = self.conv_1(img)
        #x = tf.nn.relu(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        #x = tf.nn.relu(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        #x = tf.nn.relu(x)        
        x = Flatten(x)
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        # x should be (60 x 40 x 8) here

        x = predict_transform(x, self.height, self.anchors, params.num_classes, False)
        # now it should have anchors for (60 x 40 x 2 x 8) or (60 x 40 x 16) or (4800 x 8)

        # this will filter 4800 boxes down to 1 box per object, output likely (3 x 7)
        # 8 to 7 b/c removes class probs for non predicted objects and adds class index for predicted object
        # may need to preserve all 4800 boxes for training and return indexes of predicting boxes
        x = filter_boxes(x)
        return x

    
    # 1 wide and 1 tall anchor box for each grid cell, normalize dims to fit in grid cell, grid should be 1 x 1 square
    # (x_c, y_c, width, height)
    def create_anchors(self):
        self.anchors = []
        
        wide_anchor = np.zeroes(4)
        anchor_1[0] = .5 # x center
        anchor_1[1] = .5 # y center
        anchor_1[2] = 1 # width
        anchor_1[3] = .5 # height
        self.anchors.append(wide_anchor)

        tall_anchor = np.zeroes(4)
        anchor_2[0] = .5
        anchor_2[1] = .5
        anchor_2[2] = .5
        anchor_2[3] = 1
        self.anchors.append(tall_anchor)


    @staticmethod
    def bbox_iou(box1, box2):
        box1_corner = tf.fill(tf.shape(box1), 0.0)
        box1_corner[:,0] = (box1[:,0] - box1[:,2]/2) # x center minus width/2
        box1_corner[:,1] = (box1[:,1] - box1[:,3]/2) # y center minus height/2
        box1_corner[:,2] = (box1[:,0] + box1[:,2]/2)
        box1_corner[:,3] = (box1[:,1] + box1[:,3]/2)

        box2_corner = tf.fill(tf.shape(box2), 0.0)
        box2_corner[:,0] = (box2[:,0] - box2[:,2]/2) # x center minus width/2
        box2_corner[:,1] = (box2[:,1] - box2[:,3]/2) # y center minus height/2
        box2_corner[:,2] = (box2[:,0] + box2[:,2]/2)
        box2_corner[:,3] = (box2[:,1] + box2[:,3]/2)

        #Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1_corner[:,0], box1_corner[:,1], box1_corner[:,2], box1_corner[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2_corner[:,0], box2_corner[:,1], box2_corner[:,2], box2_corner[:,3]
        
        #get the corrdinates of the intersection rectangle
        inter_rect_x1 =  tf.math.maximum(b1_x1, b2_x1)
        inter_rect_y1 =  tf.math.maximum(b1_y1, b2_y1)
        inter_rect_x2 =  tf.math.minimum(b1_x2, b2_x2)
        inter_rect_y2 =  tf.math.minimum(b1_y2, b2_y2)
        
        #Intersection area
        inter_area = tf.clip_by_value(inter_rect_x2 - inter_rect_x1 + 1, clip_value_min=0) * \
                                                    tf.clip_by_value(inter_rect_y2 - inter_rect_y1 + 1, clip_value_min=0)
    
        #Union Area
        b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
        
        iou = inter_area / (b1_area + b2_area - inter_area)
        
        return iou


    # predictions will be (4800 x 8)
    @staticmethod
    def filter_boxes(predictions):
        conf_mask = tf.Tensor(predictions[:, 4] > params.object_conf, tf.float64)
        predictions = predictions * conf_mask # this should 0 out the entire 8 vector row of 4th dim if below object threshold
        '''
        box_corner = predictions.new(predictions.shape)
        box_corner[:,0] = (predictions[:,0] - predictions[:,2]/2) # x center minus width/2
        box_corner[:,1] = (predictions[:,1] - predictions[:,3]/2) # y center minus height/2
        box_corner[:,2] = (predictions[:,0] + predictions[:,2]/2) 
        box_corner[:,3] = (predictions[:,1] + predictions[:,3]/2)
        prediction[:,:4] = box_corner[:,:4] # change from center and width/height to corner coordinates
        '''
        max_ind = tf.math.argmax(predictions[:,5:5+ params.num_classes], 1)
        max_score = tf.reduce_max(predictions[:,5:5+ params.num_classes], reduction_indices=[1])
        seq = (predictions[:,:5], max_score, max_ind) # (4800 x 5, 1, 1)
        image_pred = tf.concat(seq, 1) # 4800 x 7

        zeros = tf.zeros([4800, 1], tf.float64)
        where = tf.not_equal(image_pred, zeros)
        nonzero_indices = tf.where(where)
        try:
            image_pred_ = image_pred[nonzero_indices, :]
            image_pred_ = tf.reshape(image_pred_, [-1, 7]) # should have size(nonzero) x 7
        except:
            return # if there are no nonzero indices
        
        img_classes, _ = tf.unique(image_pred_[:,-1])
        output = None
        for cls_ind in img_classes:
            cls_mask = image_pred_*(image_pred_[:,-1] == cls_ind) # get boxes with prediction for this class
            class_mask_ind = tf.not_equal(cls_mask[:,-1], zeros)
            nonzero_indices = tf.where(class_mask_ind)
            image_pred_class = tf.reshape(image_pred_[nonzero_indices], [-1,7]) # should be boxes that have a prediction for cls_ind

            _, sorted_inds = tf.math.top_k(image_pred_class[:,4], k=tf.size(image_pred_class[:,4]))
            image_pred_class = image_pred_class[sorted_inds]
            idx = tf.size(image_pred_class)

            # NMS - multiple grid cells may detect same object. Take highest prob box and for each box that has a 
            # high IOU ( > nms_conf) with that box remove it
            zeros = tf.zeros([tf.size(image_pred_class), 1], tf.float64)            
            for i in range(idx):
                try:
                    # get iou of ith box with all boxes after it
                    ious = bbox_iou(image_pred_class[i], image_pred_class[i+1:])
                except ValueError:
                    break

                except IndexError:
                    break
                
                iou_mask = tf.Tensor(ious < params.nms_conf, tf.float64)
                image_pred_class[i+1:] *= iou_mask # zero out 7 dim vectors for boxes less than threshold
                where = tf.not_equal(image_pred_class[:,4], zeros)
                nonzero_indices = tf.where(where)
                # preserves non zero entries from previous iterations
                image_pred_class = tf.reshape(image_pred_class[nonzero_indices], [-1, 7])

            if output is None:
                output = image_pred_class
            else:
                output = tf.concat(output, image_pred_class)
        
        return output


    # boxes are stacked along depth dimension in feature map. To get the 2nd box of cell (5, 6) index it
    # by map[5, 6, (5+num_classes): 2*(5+num_classes)] (3rd dim starts after first bbox attrs and stops at start of 3rd bbox attrs)
    # this function transforms this into a 2D tensor with each row correspoding to attrs of bounding box
    # first 3 rows are for bboxes at (0,0) grid cell, next 3 rows for (0,1) cell, etc
    # this is more useful if we make detections at different scales (different number of grid boxes/strides), then all the scales
    # can be concatenated
    # However this also does some necessary ops like sigmoid and the exponential
    @staticmethod
    def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
        batch_size = prediction.size(0)
        stride =  inp_dim // prediction.size(2)
        grid_size = inp_dim // stride
        bbox_attrs = 5 + num_classes
        num_anchors = len(anchors)

        # prediction is (grid_cell_x, grid_cell_y, 5 + num_classes)
        prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
        prediction = prediction.transpose(1,2).contiguous()
        prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

        # makes the x_center coordinate between 0 and 1, each grid cell is normalized to 1 x 1
        prediction[:,:,0] = tf.math.sigmoid(prediction[:,:,0])
        # makes the y_center coordinate between 0 and 1, each grid cell is normalized to 1 x 1
        prediction[:,:,1] = tf.math.sigmoid(prediction[:,:,1])
        # makes the objectness score a probability between 0 and 1
        prediction[:,:,4] = tf.math.sigmoid(prediction[:,:,4])

        # np.arrange creates an array of ints from 0 to grid_size - 1
        grid = np.arange(grid_size)
        # this creates 2D array corresponding to our grid over the image
        a,b = np.meshgrid(grid, grid)


        x_offset = tf.Tensor(a, tf.float64)
        # this makes it a 2D array with rows and cols instead of 1D array
        # so there is now an x_offset corresponding to each grid cell in 2D grid
        x_offset = tf.reshape(x_offset, [-1, -1])
        y_offset = tf.Tensor(b, tf.float64)
        y_offset = tf.reshape(y_offset, [-1, -1])

        if CUDA:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        # concatenates the grids horizontally
        x_y_offset = tf.concat((x_offset, y_offset), 1)
        # makes num_anchors copies and appends vertically, may need to use tile instead
        x_y_offset = tf.repeat(x_y_offset, repeats=[num_anchors, num_anchors], axis=0)
        # unflattens so now we have array for each grid cell, [(0,0), (0,1), ...]
        x_y_offset = tf.reshape(x_y_offset, [-1, 2])
        # inserts dimension with size 1, doesn't change arrangement of tensor
        x_y_offset = tf.expand_dims(x_y_offset, 0)
        # sets the first elems of each (5 + num_classes) array in predictions to center coordinates we just arranged
        prediction[:,:,:2] += x_y_offset

        # convert list of 3 anchors to tensor
        anchors = tf.Tensor(anchors, tf.float64)
        if CUDA:
            anchors = anchors.cuda()
        anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
        # every grid cell gets a copy of the anchors
        anchors = tf.repeat(anchors, repeats=[grid_size*grid_size, grid_size*grid_size], axis=0)
        anchors = tf.expand_dims(anchors, 0)
        # apply exponential transformation to all predicted widths and heights and multiply by anchors
        # this equation is here https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
        prediction[:,:,2:4] = tf.math.exp(prediction[:,:,2:4])*anchors

        # apply sigmoid to class scores to make them probabilities
        prediction[:,:,5: 5 + num_classes] = tf.math.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

        # resize coordinates to size of input image, they were normalized to 1 x 1 grid cell boxes
        prediction[:,:,:4] *= stride

        return prediction # goal of this function is to return (4800 x 8)