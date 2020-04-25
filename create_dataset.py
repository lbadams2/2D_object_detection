from waymo_open_dataset import dataset_pb2 as open_dataset
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


def draw_image_with_classes(image, labels, classes, num, trues=None):
    colors = ['black', 'red', 'green', 'blue', 'yellow']
    for vec, cls in zip(labels, classes):
        print(cls, type(cls))
        plt.gca().add_patch(patches.Rectangle(
          xy=(vec[0] - 0.5 * vec[2], vec[1] - 0.5 * vec[3]),
          width=vec[2],
          height=vec[3],
          linewidth=1.5,
          edgecolor=colors[int(cls)],
          facecolor='none'))
    if trues != None:
      for vec in trues:
        plt.gca().add_patch(patches.Rectangle(
          xy=(vec[0] - 0.5 * vec[2], vec[1] - 0.5 * vec[3]),
          width=vec[2],
          height=vec[3],
          linewidth=1.5,
          edgecolor='yellow',
          facecolor='none'))
    image = tf.image.resize(image, (params.scaled_height, params.scaled_width))
    plt.imshow(image)
    plt.grid(False)
    plt.axis('off')
    plt.savefig('testing/new/test-{}.jpeg'.format(num))
    plt.clf()


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



def create_true_box_grid_validation(y):
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
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0) # will be positive, subtracting negative
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        # its possible 2 objects in same cell have same anchor, last will overwrite
        if best_iou > 0:
            box_coords = box[:4]
            box_coords = box_coords / np.array([params.grid_width, params.grid_height, params.grid_width, params.grid_height])
            box_coords = box_coords * np.array([params.scaled_width, params.scaled_height, params.scaled_width, params.scaled_height])
            center_coords = box_coords[:2]
            wh_coords = box_coords[2:4]
            box_mins = center_coords - (wh_coords / 2.)
            box_maxes = center_coords + (wh_coords / 2.)
            corner_boxes = tf.concat([box_mins, box_maxes], axis=-1)
            adjusted_box = np.array(
                [
                    corner_boxes[0],
                    corner_boxes[1],
                    corner_boxes[2], # quotient might be less than one, not sure why log is used
                    corner_boxes[3],
                    box[4] # class label
                ],
                dtype=np.float32
            )
            true_box_grid[i, j, best_anchor] = adjusted_box
            box_mask[i, j, best_anchor] = mask_vec
    # could pad to 1920 x 1920 with 0s then resize to 416 x 416
    
    return true_box_grid, box_mask



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
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0) # will be positive, subtracting negative
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
                    box[0] - j, # center should be between 0 and 1, like prediction will be
                    box[1] - i,
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


def image_example(image_string, obj_vectors, training):

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
    if training:
        true_box_grid, box_mask = create_true_box_grid(objs)
    else:
        true_box_grid, box_mask = create_true_box_grid_validation(objs)
    #print('grid shape ', true_box_grid.shape)
    true_box_grid_flat = true_box_grid.reshape(-1)
    #print('flattened shape ', true_box_grid_flat.shape)
    #print('')
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
    #if len(obj_vectors) > 0:
    #    print(example)
    return example


# breaking on file segment-17539775446039009812_440_000_460_000_with_camera_labels.tfrecord
# standard jpeg dimensions are width by height, images in waymo dataset seem to be height by width after tf decode
def convert(training, record_file, path):

    images = []
    img_boxes = []
    count = 0    
    examples = []
    img_num = 0
    max_x = 0
    max_y = 0
    break_var = False
    for filename in os.listdir(path):
        if break_var:
            break
        filepath = path + filename
        dataset = tf.data.TFRecordDataset(filepath, compression_type='')
        for data in dataset:
            if break_var:
                break
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
                if img_num == 500 and not training:
                    print('breaking on file', filename)
                    break_var = True
                    break
                
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
                        l[4] = params.TYPE_VEHICLE
                    elif label.type == 2:
                        l[4] = params.TYPE_PEDESTRIAN
                    elif label.type == 3:
                        l[4] = params.TYPE_SIGN
                    elif label.type == 4:
                        l[4] = params.TYPE_CYCLIST
                    else:
                        l[4] = params.TYPE_UNKNOWN
                    obj_vectors.append(l)
                
                #draw_orig_image(image, obj_vectors, count)
                count += 1
                ex = image_example(image_obj.image, obj_vectors, training)
                examples.append(ex)

    #print('max x ', max_x)    
    #print('max y', max_y)
    with tf.io.TFRecordWriter(record_file) as writer:
        for ex in examples:            
            writer.write(ex.SerializeToString())   

    #print('outer for loop {} times'.format(count))
    #y = tf.ragged.constant(img_boxes)
    #new_dataset = tf.data.Dataset.from_tensor_slices((images, y))



def draw_image(image, labels, num):
    #image = tf.image.decode_jpeg(image)
    #print(image)
    #vecs = tf.sparse.to_dense(labels)
    #if len(vecs.shape) > 2:
    #    vecs = tf.squeeze(vecs, 0)
    vecs = labels
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


def test_format_data(image, true_box_grid, mask_grid, labels):
    image = tf.image.decode_jpeg(image)
    true_box_grid = tf.reshape(true_box_grid, [params.grid_height, params.grid_width, params.num_anchors, params.true_vec_len])
    mask_grid = tf.reshape(mask_grid, [params.grid_height, params.grid_width, params.num_anchors, 2])
    vecs = tf.sparse.to_dense(labels)
    return image, true_box_grid, mask_grid, vecs


def gather_grid_boxes(dataset):
    # add grid offsets, divide by conv_size, then multiply by orig size to get original coord values
    conv_size = np.array([params.grid_width, params.grid_height, params.grid_width, params.grid_height])
    orig_size = np.array([params.im_width, params.im_height, params.im_width, params.im_height])
    anchors = DetectNet.get_anchors()
    count = 0
    for img, true_box_grid, box_mask, true_boxes in dataset:        
        num_objs = true_boxes.shape[0]
        count += 1
        if num_objs == 0:
            continue
        orig_boxes = true_boxes[:,:4] / orig_size
        orig_boxes = orig_boxes * conv_size
        i = np.floor(orig_boxes[:,1]).astype('int') # row, y coord
        j = np.floor(orig_boxes[:,0]).astype('int') # column, x coord
        if num_objs > 1:
            zero = tf.zeros_like(true_box_grid)
            # where will be same shape as true_box_grid with true or false in each cell if not equal to zero
            # can use []
            where = tf.not_equal(true_box_grid, zero)            
            indices = tf.where(where)
            grid_indices = indices[:,:3] # grid and anchors of boxes, will be duplicates for each vec elem
            #t1d = tf.reshape(grid_indices, shape=(-1,))
            #uniques, idx, counts = tf.unique_with_counts(t1d)
            #idx = tf.reshape(idx, shape=tf.shape(grid_indices)) # don't need this
            uniques = np.unique(grid_indices.numpy(), axis=0) # tensorflow can't do this
            uniques = tf.convert_to_tensor(uniques)

            grid_offsets = uniques[:,:2]
            grid_offsets = tf.cast(grid_offsets, tf.float32)
            anchor_idx = uniques[:,2]
            # grid_vec coords are (j, i)/(x,y) tensor dims are (i, j)/(y,x)
            x_offset = grid_offsets[:,1]
            y_offset = grid_offsets[:,0]
            x_offset = tf.expand_dims(x_offset, 1)
            y_offset = tf.expand_dims(y_offset, 1)
            grid_offsets = tf.concat([x_offset, y_offset], axis=1)
            # gather indices of true box grid
            grid_vectors = tf.gather_nd(true_box_grid, uniques)            
            center_coords = grid_vectors[:,:2]
            center_coords = center_coords + grid_offsets
            wh_coords = grid_vectors[:,2:4]
            anchor = tf.gather(anchors, anchor_idx)            
            wh_coords = tf.math.exp(wh_coords)
            wh_coords = wh_coords * anchor
            if len(center_coords.shape) < 2:
                center_coords = tf.expand_dims(center_coords, 0)
                wh_coords = tf.expand_dims(wh_coords, 0)
            orig_grid_vectors = tf.concat([center_coords, wh_coords], axis=1)
            orig_grid_vectors = orig_grid_vectors / conv_size
            orig_grid_vectors = orig_grid_vectors * orig_size
            draw_image(img, orig_grid_vectors, count)
        else:
            continue
            #print('no objects')


def test_grid_dataset():
    dataset = tf.data.TFRecordDataset('image_grid_dataset_train.tfrecord')
    dataset = dataset.map(train._parse_image_function)
    dataset = dataset.map(test_format_data)
    gather_grid_boxes(dataset)

    '''
    # need to draw image before format data
    i = 0
    for img, _, _, true_boxes in dataset:
        draw_image(img, true_boxes, i)
        i += 1
    dataset = dataset.map(train.format_data)
    '''
    #dataset = dataset.map(train.remove_orig_boxes)

    '''
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
    '''



if __name__ == '__main__':
    convert(False, 'validation.tfrecord', 'data/validation/') # this creates the tfrecord file
    #test_grid_dataset()
    #convert_test()