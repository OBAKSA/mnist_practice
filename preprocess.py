import tensorflow as tf
import numpy as np
import glob
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def _bytes_feature(value):
    """ Returns a bytes_list from a string/byte"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """ Returns a float_list from a float/double """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """ Returns a int64_list from a bool/enum/int/uint """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(img, lab):
    """ TODO: Return serialized Example from img, lab """
    pass


def write_tfrecord(imagedir, datadir):
    """ TODO: write a tfrecord file containing img-lab pairs
        imagedir: directory of input images
        datadir: directory of output a tfrecord file (or multiple tfrecord files) """
    pass


def check(filename):
    iterator = tf.python_io.tf_record_iterator(path=filename)

    for i in range(3):
        serialized_record = next(iterator)

        example = tf.train.Example()
        example.ParseFromString(serialized_record)

        img = example.features.feature['encoded'].bytes_list.value
        lab = example.features.feature['label'].int64_list.value[0]

        img = np.fromstring(img[0], dtype=np.uint8)
        img = img.reshape([28,28])
        plt.imshow(img)
        plt.show()

# def read_tfrecord(folder, batch=16, epoch=1):
#     """ TODO: read tfrecord files in folder, Return shuffled mini-batch img,lab pairs
#     folder: directory where tfrecord files are stored in
#     epoch: maximum epochs to train, default: 1 """
#     pass
def read_tfrecord(folder, batch=16, epoch=1):
    # filename queue
    filenames = glob.glob(folder+'/*.tfrecord')
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epoch)

    # read serialized examples
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    # parse examples into features, each
    key_to_feature = {'encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                      'label': tf.FixedLenFeature([], tf.int64, default_value=0)}

    features = tf.parse_single_example(serialized_example, features=key_to_feature)

    # decode data
    img = tf.decode_raw(features['encoded'], tf.uint8)
    img = tf.reshape(img, [28,28,1])
    img = tf.cast(img, tf.float32)
    img /= 255
    lab = tf.cast(features['label'], tf.int32)
    lab = tf.one_hot(lab, 10)

    # mini-batch examples queue
    batch_size = batch
    min_after_dequeue = 10

    img, lab = tf.train.shuffle_batch([img, lab], batch_size=batch_size,
                                      capacity=min_after_dequeue+3*batch_size, num_threads=1,
                                      min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=False)
                                            # capacity=min_after_dequeue+3*batch_size, num_threads=1,
                                            # min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=False)

    return img, lab
