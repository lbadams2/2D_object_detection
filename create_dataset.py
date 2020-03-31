from waymo_open_dataset import dataset_pb2 as open_dataset
import numpy as np
import tensorflow as tf
import params
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

plt.figure(figsize=(16, 8))


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
    plt.savefig('testing/orig/test-{}.png'.format(num))
    plt.clf()



def convert():
    record_file = 'image_dataset_train.tfrecords'

    images = []
    img_boxes = []
    count = 0    
    examples = []
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
                if image.shape[0] != 1280 or image.shape[1] != 1920:
                    continue            
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
                
                #draw_orig_image(image, obj_vectors, count)
                count += 1
                ex = image_example(image_obj.image, obj_vectors)
                examples.append(ex)
        
    with tf.io.TFRecordWriter(record_file) as writer:
        for ex in examples:            
            writer.write(ex.SerializeToString())            

    #print('outer for loop {} times'.format(count))
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


def draw_image(image, labels, num):
    vecs = labels['Box Vectors']
    image = image['image']
    image = tf.reshape(image, [])
    #print(image)
    image = tf.image.decode_jpeg(image)
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
    plt.savefig('testing/new/test-{}.png'.format(num))
    plt.clf()


def test_new_dataset():
    image_dataset = tf.data.TFRecordDataset('image_dataset_train.tfrecords')
    parsed_image_dataset = image_dataset.map(_parse_image_function)
    print('size of dataset ', len(list(parsed_image_dataset)))
    #parsed_image_dataset = parsed_image_dataset.shuffle(100)
    #batched_ds = parsed_image_dataset.batch(1)
    i = 0
    for image, labels in parsed_image_dataset:
        if i % 500 == 0:
            draw_image(image, labels, i)
        i += 1


if __name__ == '__main__':
    convert() # this creates the tfrecord file
    #test_new_dataset() # this will sample the new file and draw some images and boxes