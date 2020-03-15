from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras import layers
import tensorflow as tf
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

Use 3 anchor boxes, 1 bottom half of the image, 1 left half, and 1 right half
'''



# each grid cell has 3 boxes associated with it for (3 x (5 + num_classes)) entires in feature map
# 5 is 4 coordinates and objectness score
class DetectNet(layers.Layer):
    def __init__(self, width, height):
        super(DetectNet, self).__init__()
        self.width = width
        self.height = height

        self.conv_1 = Conv2D(32, 6, 6, input_shape=X.shape[1:], dim_ordering='tf', activation='relu')
        self.pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))
        self.conv_2 = Conv2D(64, filter_size, filter_size, dim_ordering='tf', activation='relu')
        self.pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))
        self.conv_3 = Conv2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu')
        self.dropout = Dropout(0.4)
        self.linear_1 = Dense(256, activation='relu')
        self.linear_2 = Dense(7 * boxes_per_img * boxes_per_img)

        self.anchors = self.create_anchors(im_height, im_width)

    def call(self, inputs):
        x = self.conv_1(inputs)
        #x = tf.nn.relu(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        #x = tf.nn.relu(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        #x = tf.nn.relu(x)
        x = predict_transform(x, self.height, self.anchors, num_classes, False)
        x = Flatten(x)
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

    
    # almost all objects seem to be in bottom half of image
    # (x_c, y_c, width, height)
    def create_anchors(self, im_height, im_width):
        self.anchors = []
        anchor_height = int(im_height * .66)
        anchor_width = int(im_width / 2)
        y_c = anchor_height / 2
        
        anchor_1 = np.zeroes(4)
        x_c1 = im_width / 4
        anchor_1[0] = x_c1
        anchor_1[1] = y_c
        anchor_1[2] = anchor_width
        anchor_1[3] = anchor_height
        self.anchors.append(anchor_1)

        anchor_2 = np.zeroes(4)
        x_c2 = im_width / 2
        anchor_1[0] = x_c2
        anchor_1[1] = y_c
        anchor_1[2] = anchor_width
        anchor_1[3] = anchor_height
        self.anchors.append(anchor_2)

        anchor_3 = np.zeroes(4)
        x_c3 = (im_width / 4) * 3
        anchor_1[0] = x_c3
        anchor_1[1] = y_c
        anchor_1[2] = anchor_width
        anchor_1[3] = anchor_height
        self.anchors.append(anchor_3)



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

        return prediction