import numpy as np

########## Hyper Parameters ##################
batch_size = 16
epochs = 10
kernel_size = 3
pool_size = 2
grid_stride = 32 # could also use 40
object_conf = .6
nms_conf = .5
coord_loss_weight = 5
noobj_loss_weight = 1
num_anchors = 5
max_boxes = 10
scaled_height = 416
scaled_width = 416

# for 13 x 13 grid - (416 / 32)
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))
##############################################


############ Constants #############
im_height = 1280 # rows
im_width = 1920 # columns
grid_height = int(scaled_height // grid_stride)
grid_width = int(scaled_width // grid_stride)
num_classes = 5
vec_len = 5 + num_classes
####################################