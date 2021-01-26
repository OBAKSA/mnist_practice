import tensorflow as tf
import numpy as np
import glob
import os
# use following commands when 'Segmentation fault' error occurs
# import matplotlib
# matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
from PIL import Image


def _bytes_feature(value):
    """ Returns a bytes_list from a string/byte"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """ Returns a float_list from a float/double """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """ Returns a int64_list from a bool/enum/int/uint """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _image_as_bytes(imagefile):
    image = np.array(Image.open(imagefile))
    image_raw = image.tostring()
    return image_raw


def make_example(img, lab):
    """ TODO: Return serialized Example from img, lab """
    feature = {'encoded': _bytes_feature(img),
               'label': _int64_feature(lab)}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()


def write_tfrecord(imagedir, datadir):
    """ TODO: write a tfrecord file containing img-lab pairs
        imagedir: directory of input images
        datadir: directory of output a tfrecord file (or multiple tfrecord files) """
    filedirs = os.listdir(imagedir)
    writer = tf.python_io.TFRecordWriter(datadir)

    for filedir in filedirs:
        filenames = os.listdir(imagedir + '/' + filedir)
        for filename in filenames:
            #img_data = open(imagedir + '/' + filedir + '/' + filename, 'rb').read()
            img_data = _image_as_bytes(imagedir + '/' + filedir + '/' + filename)
            lab = int(filedir)

            example = make_example(img_data, lab)
            writer.write(example)

    writer.close()


def read_tfrecord(folder, batch=100, epoch=1):
    """ TODO: read tfrecord files in folder, Return shuffled mini-batch img,lab pairs
    img: float, 0.0~1.0 normalized
    lab: dim 10 one-hot vectors
    folder: directory where tfrecord files are stored in
    epoch: maximum epochs to train, default: 1 """
    filenames_temp = os.listdir(folder)
    filenames = []
    for filename_temp in filenames_temp:
        filename = folder + '/' + filename_temp
        filenames.append(filename)

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epoch)

    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    key_to_feature = {'encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                      'label': tf.FixedLenFeature([], tf.int64, default_value=0)}

    features = tf.parse_single_example(serialized_example, features=key_to_feature)

    img = tf.decode_raw(features['encoded'], tf.uint8)
    img = tf.reshape(img, [28, 28, 1])
    img = tf.cast(img, tf.float32)
    #img = tf.cast(img, tf.float32)/tf.norm(img)
    img = tf.div(tf.subtract(img, tf.reduce_min(img)), tf.subtract(tf.reduce_max(img), tf.reduce_min(img)))

    lab = tf.cast(features['label'], tf.int32)
    lab = tf.constant(np.eye(10))[lab]

    batch_size = batch
    min_after_dequeue = 5000

    img, lab = tf.train.shuffle_batch([img, lab], batch_size=batch_size,
                                      capacity=min_after_dequeue + 3 * batch_size, num_threads=1,
                                      min_after_dequeue=min_after_dequeue)

    return img, lab
