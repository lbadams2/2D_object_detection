########## Hyper Parameters ##################
batch_size = 16
epochs = 10
kernel_size = 3
pool_size = 2
grid_stride = 32 # could also use 40
object_conf = .6
nms_conf = .5
coord_loss_weight = 5
noobj_loss_weight = .5
num_anchors = 2
max_boxes = 20 # max boxes to return from nms
##############################################


############ Constants #############
im_height = 1280 # rows
im_width = 1920 # columns
num_classes = 5
vec_len = 5 + num_classes
####################################