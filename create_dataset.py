from waymo_open_dataset import dataset_pb2 as open_dataset
import numpy as np
import tensorflow as tf
import params
import matplotlib.pyplot as plt
import matplotlib.patches as patches

'''
FILENAME = 'data/train/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
#num = sum(1 for _ in dataset)

num = 0
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    for camera_labels in frame.camera_labels:
        num += 1

print('size of data set is ', num)
'''
record_file = 'image_dataset.tfrecords'

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

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
    image = tf.train.Features(feature={
        'image': _bytes_feature(image_string),
    })
    example = tf.train.SequenceExample(
        context=image,
        feature_lists=boxes
    )
    return example


def convert():
    images = []
    img_boxes = []
    i = 0
    FILENAME = 'data/train/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    for data in dataset:        
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        #print('num images in frame {}, num labels in frame {}'.format(len(frame.images), len(frame.camera_labels)))
        for camera_labels in frame.camera_labels:
            camera_name = camera_labels.name # 0 to 5
            image_obj = next((x for x in frame.images if x.name == camera_name), None)
            image = tf.image.decode_jpeg(image_obj.image)
            if image.shape[0] != 1280 or image.shape[1] != 1920:
                continue
            i += 1
            num_objects = len(camera_labels.labels)
            obj_vectors = []
            for i in range(num_objects):
                label = camera_labels.labels[i]
                l = [0] * 10
                l[0] = label.box.center_x
                l[1] = label.box.center_y
                l[2] = label.box.width
                l[3] = label.box.length
                l[4] = 1 # objectness score
                if label.type == 'TYPE_PEDESTRIAN':
                    l[5] = 1
                elif label.type == 'TYPE_VEHICLE':
                    l[6] = 1
                elif label.type == 'TYPE_CYCLIST':
                    l[7] = 1
                elif label.type == 'TYPE_SIGN':
                    l[8] = 1
                else:
                    l[9] = 1
                obj_vectors.append(l)

            with tf.io.TFRecordWriter(record_file) as writer:
                ex = image_example(image_obj.image, obj_vectors)
                writer.write(ex.SerializeToString())                

    print('outer for loop {} times'.format(i))
    #y = tf.ragged.constant(img_boxes)
    #new_dataset = tf.data.Dataset.from_tensor_slices((images, y))

def _parse_image_function(example):
    # Create a dictionary describing the features.
    context_feature = {
        'image': tf.io.FixedLenFeature([], dtype=tf.string)
    }
    sequence_features = {
        'Box Vectors': tf.io.VarLenFeature(dtype=tf.float32)
    }
    context_data, sequence_data = tf.io.parse_single_sequence_example(serialized=example, 
                                    context_features=context_feature, sequence_features=sequence_features)

    # Parse the input tf.Example proto using the dictionary above.
    return context_data, sequence_data

def draw_image(image, labels):
    vecs = labels['Box Vectors']
    image = image['image']
    image = tf.reshape(image, [])
    #print(image)
    image = tf.image.decode_jpeg(image)
    plt.figure(figsize=(16, 8))
    vecs = tf.sparse.to_dense(vecs)
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
    plt.savefig('test.png')


def test_new_dataset():
    image_dataset = tf.data.TFRecordDataset('image_dataset.tfrecords')
    print('size of dataset ', len(list(image_dataset)))
    parsed_image_dataset = image_dataset.map(_parse_image_function)
    print('size of dataset ', len(list(parsed_image_dataset)))
    #parsed_image_dataset = parsed_image_dataset.shuffle(100)
    #batched_ds = parsed_image_dataset.batch(1)
    i = 1
    for image, labels in parsed_image_dataset:
        #print(labels['Box Vectors'])
        print('1 iteration')
        if i % 10 == 0:
            draw_image(image, labels)
            break
        i += 1


convert()
#test_new_dataset()
        