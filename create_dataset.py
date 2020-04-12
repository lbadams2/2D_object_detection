#from waymo_open_dataset import dataset_pb2 as open_dataset
import numpy as np
import tensorflow as tf
import params
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from detect_net import DetectNet
import train

plt.figure(figsize=(16, 8))


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def draw_orig_image(image, labels, num):    
    for vec in labels:
        plt.gca().add_patch(patches.Rectangle(
          xy=(vec[0] - 0.5 * vec[2], vec[1] - 0.5 * vec[3]),
          width=vec[2],
          height=vec[3],
          linewidth=1,
          edgecolor='red',
          facecolor='none'))
    
    #print(image.shape)
    plt.imshow(image)
    plt.grid(False)
    plt.axis('off')
    plt.savefig('testing/new/test-{}.jpeg'.format(num))
    plt.clf()



# tensors are (rows, cols)
# coords are (x, y) - opposite
# create class mask here too
def create_true_box_grid(y):
    #y = y.numpy()
    true_box_grid = np.zeros(shape=(params.grid_height, params.grid_width, params.num_anchors, params.true_vec_len), dtype=np.float32)
    box_mask = np.zeros(shape=(params.grid_height, params.grid_width, params.num_anchors, 2), dtype=np.float32)
    mask_vec = np.ones((2), dtype=np.float32)
    if y.shape[0] == 0:
        return true_box_grid, box_mask
    
    #print('found objects')
    orig_size = np.array([params.im_width, params.im_height])
    center_boxes = y[:,:2] / orig_size
    #center_boxes = tf.cast(center_boxes, tf.float32)
    boxes_wh = y[:,2:4] / orig_size
    #boxes_wh = tf.cast(boxes_wh, tf.float32)
    conv_size = np.array([params.grid_width, params.grid_height])
    #conv_size = tf.cast(conv_size, tf.float32)
    center_boxes = center_boxes * conv_size
    boxes_wh = boxes_wh * conv_size
    # box coords are now normalized to be between [grid_width, grid_height] (same scale as anchors)
    true_boxes = np.concatenate([center_boxes, boxes_wh, y[:,4:]], axis=1)
    anchors = DetectNet.get_anchors().numpy()
    
    num_objs = y.shape[0]
    for obj in range(num_objs):
        best_iou = 0
        best_anchor = 0
        box = true_boxes[obj]
        i = np.floor(box[1]).astype('int') # row
        j = np.floor(box[0]).astype('int') # column
        #j = tf.cast(j, tf.float32) 

        # find best anchor for current box
        for k, anchor in enumerate(anchors):
            # center box
            box_max = box[2:4] / 2
            box_min = -box_max

            # center anchor
            anchor_max = anchor / 2
            anchor_min = -anchor_max

            intersect_mins = np.maximum(box_min, anchor_min)
            intersect_maxes = np.minimum(box_max, anchor_max)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        # its possible 2 objects in same cell have same anchor, last will overwrite
        if best_iou > 0:
            adjusted_box = np.array(
                [
                    box[0] - i, # center should be between 0 and 1, like prediction will be
                    box[1] - j,
                    np.log(box[2] / anchors[best_anchor][0]), # quotient might be less than one, not sure why log is used
                    np.log(box[3] / anchors[best_anchor][1]),
                    box[4] # class label
                ],
                dtype=np.float32
            )
            true_box_grid[i, j, best_anchor] = adjusted_box
            box_mask[i, j, best_anchor] = mask_vec
        
    # could pad to 1920 x 1920 with 0s then resize to 416 x 416
    
    return true_box_grid, box_mask


def image_example(image_string, obj_vectors):

    box_feature_list = []
    for vec in obj_vectors:
        box_features = tf.train.Feature(float_list=tf.train.FloatList(value=vec))
        box_feature_list.append(box_features)

    all_box_features = tf.train.FeatureList(feature=box_feature_list)
    box_dict = {
        'Box Vectors': all_box_features
    }
    boxes = tf.train.FeatureLists(feature_list=box_dict)

    objs = np.array(obj_vectors)
    true_box_grid, box_mask = create_true_box_grid(objs)
    print('grid shape ', true_box_grid.shape)
    true_box_grid_flat = true_box_grid.reshape(-1)
    print('flattened shape ', true_box_grid_flat.shape)
    print('')
    true_box_mask_flat = box_mask.reshape(-1)

    box_grid_feature = tf.train.Feature(float_list=tf.train.FloatList(value=true_box_grid_flat))
    box_mask_feature = tf.train.Feature(float_list=tf.train.FloatList(value=true_box_mask_flat))

    fixed_features = tf.train.Features(feature={
        'image': _bytes_feature(image_string),
        'true_grid': box_grid_feature,
        'mask_grid': box_mask_feature,
    })
    example = tf.train.SequenceExample(
        context=fixed_features,
        feature_lists=boxes
    )
    if len(obj_vectors) > 0:
        print(example)
    return example


# standard jpeg dimensions are width by height, images in waymo dataset seem to be height by width after tf decode
def convert():
    record_file = 'image_grid_dataset_train.tfrecord'

    images = []
    img_boxes = []
    count = 0    
    examples = []
    img_num = 0
    max_x = 0
    max_y = 0
    for filename in os.listdir('data/train'):
        filepath = 'data/train/' + filename
        dataset = tf.data.TFRecordDataset(filepath, compression_type='')
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            #print('num images in frame {}, num labels in frame {}'.format(len(frame.images), len(frame.camera_labels)))
            for camera_labels in frame.camera_labels:
                camera_name = camera_labels.name # 0 to 5
                image_obj = next((x for x in frame.images if x.name == camera_name), None)                
                image = tf.image.decode_jpeg(image_obj.image)                
                #print('image shape {} {}'.format(image.shape[0], image.shape[1]))
                
                if image.shape[0] != params.im_height or image.shape[1] != params.im_width:
                    continue
                if img_num % 100 == 0:
                    print('processing image {}'.format(img_num))
                    #print('max x {} max y {}'.format(max_x, max_y))
                img_num += 1
                
                num_objects = len(camera_labels.labels)
                obj_vectors = []
                for i in range(num_objects):
                    label = camera_labels.labels[i]
                    l = [0] * params.true_vec_len
                    l[0] = label.box.center_x
                    if label.box.center_x > max_x:
                        max_x = label.box.center_x
                    l[1] = label.box.center_y
                    if label.box.center_y > max_y:
                        max_y = label.box.center_y
                    l[2] = label.box.width
                    l[3] = label.box.length
                    #l[4] = 1 # objectness score
                    if label.type == 1:
                        l[4] = params.TYPE_PEDESTRIAN
                    elif label.type == 2:
                        l[4] = params.TYPE_VEHICLE
                    elif label.type == 3:
                        l[4] = params.TYPE_CYCLIST
                    elif label.type == 4:
                        l[4] = params.TYPE_SIGN
                    else:
                        l[4] = params.TYPE_UNKNOWN
                    obj_vectors.append(l)
                
                #draw_orig_image(image, obj_vectors, count)
                count += 1
                ex = image_example(image_obj.image, obj_vectors)
                examples.append(ex)

    print('max x ', max_x)    
    print('max y', max_y)
    with tf.io.TFRecordWriter(record_file) as writer:
        for ex in examples:            
            writer.write(ex.SerializeToString())   

    #print('outer for loop {} times'.format(count))
    #y = tf.ragged.constant(img_boxes)
    #new_dataset = tf.data.Dataset.from_tensor_slices((images, y))



def draw_image(image, labels, num):
    image = tf.image.decode_jpeg(image)
    #print(image)
    vecs = tf.sparse.to_dense(labels)
    if len(vecs.shape) > 2:
        vecs = tf.squeeze(vecs, 0)

    for i in range(vecs.shape[0]):
        vec = vecs[i].numpy()
        plt.gca().add_patch(patches.Rectangle(
          xy=(vec[0] - 0.5 * vec[2], vec[1] - 0.5 * vec[3]),
          width=vec[2],
          height=vec[3],
          linewidth=1,
          edgecolor='red',
          facecolor='none'))
    
    #print(image.shape)
    plt.imshow(image)
    plt.grid(False)
    plt.axis('off')
    plt.savefig('testing/new/test-{}.png'.format(num))
    plt.clf()



def convert_to_string(image):
    image = tf.reshape(image, [])
    print(image)


def test_grid_dataset():
    dataset = tf.data.TFRecordDataset('image_grid_dataset_train.tfrecord')
    dataset = dataset.map(train._parse_image_function)
    # need to draw image before format data
    i = 0
    for img, _, _, true_boxes in dataset:
        draw_image(img, true_boxes, i)
        i += 1
    dataset = dataset.map(train.format_data)
    
    #dataset = dataset.map(train.remove_orig_boxes)

    i = 0
    for img, true_box_grid, box_mask, true_boxes in dataset:
        num_objs = true_boxes.shape[0]
        bool_mask = tf.concat([box_mask, box_mask, box_mask], axis=3)
        bool_mask = tf.cast(bool_mask, tf.bool)
        bool_mask = bool_mask[...,:5]
        grid_box_filtered = tf.boolean_mask(true_box_grid, bool_mask)
        num_grid_objs = tf.size(grid_box_filtered).numpy() / 5
        num_true_objs = tf.size(true_boxes).numpy() / 10
        if num_grid_objs != num_true_objs:
            print('objs not equal {} {} {}'.format(num_grid_objs, num_true_objs, i))
        draw_orig_image(img, true_boxes, i)
        i += 1



if __name__ == '__main__':
    convert() # this creates the tfrecord file
    #test_grid_dataset()
    #convert_test()