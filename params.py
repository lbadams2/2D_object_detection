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
num_anchors = 2
img_scale_factor = 13 / 40 # (416 x 624)
##############################################


############ Constants #############
im_height = 1280 # rows
im_width = 1920 # columns
scaled_height = im_height * img_scale_factor
scaled_width = im_width * img_scale_factor
grid_height = scaled_height / grid_stride
grid_width = scaled_width / grid_stride
num_classes = 5
vec_len = 5 + num_classes
####################################