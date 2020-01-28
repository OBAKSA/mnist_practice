import tensorflow as tf
import numpy as np
import glob
import os
# import ipdb
import matplotlib.pyplot as plt
# import PIL.Image as Image
# import keras
# import cv2
# import skimage



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(img, lab, h, w, filename):
    feature = {'encoded': _bytes_feature(img),
               'label': _float_feature(lab),
               'height': _int64_feature(h),
               'width': _int64_feature(w)}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()


def write_tfrecord():
    filenames = glob.glob('hotdog/*')

    writer = tf.python_io.TFRecordWriter('Example.tfrecord')

    for i in range(100):
        filename = filenames[i]
        img = plt.imread(filename)
        h, w, _ = img.shape
        img_data = open(filename, 'rb').read()
        lab = i*0.5

        example = make_example(img_data, lab, h, w, filename)
        writer.write(example)
    writer.close()


def check(filename):
    iterator = tf.python_io.tf_record_iterator(path=filename)

    for i in range(3):
        serialized_record = next(iterator)
    
        example = tf.train.Example()
        example.ParseFromString(serialized_record)
    
        img = example.features.feature['encoded'].bytes_list.value
        lab = example.features.feature['label'].float_list.value[0]
        h = example.features.feature['height'].int64_list.value[0]
        w = example.features.feature['width'].int64_list.value[0]
        filename = example.features.feature['filename'].bytes_list.value
        
        print('lab: %f, h: %d, w: %d, filename: %s' % (lab, h, w, filename))
        img = np.fromstring(img[0], dtype=np.uint8)
        img = img.reshape([84,84,3])
        plt.imshow(img)
        plt.show()


def read_tfrecord():
    # filename queue
    filenames = glob.glob('*.tfrecord')
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=5)

    # read serialized examples
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    # parse examples into features, each
    key_to_feature = {'encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                      'label': tf.FixedLenFeature([], tf.float32, default_value=0.),
                      'height': tf.FixedLenFeature([], tf.int64, default_value=0),
                      'width': tf.FixedLenFeature([], tf.int64, default_value=0)}

    features = tf.parse_single_example(serialized_example, features=key_to_feature)

    # decode data
    img = tf.decode_raw(features['encoded'], tf.uint8)
    img_shape = tf.shape(img)
    img = tf.reshape(img, [84,84,3])
    h = tf.cast(features['height'], tf.int32)
    w = tf.cast(features['width'], tf.int32)
    lab = tf.cast(features['label'], tf.float32)

    # mini-batch examples queue
    batch_size = 5
    min_after_dequeue = 10

    img, h, w, lab = tf.train.shuffle_batch([img, h, w, lab], batch_size=5,
                                            capacity=min_after_dequeue+3*batch_size, num_threads=1,
                                            min_after_dequeue=min_after_dequeue)

    return img, lab, h, w, img_shape


def main():
    write_tfrecord()
    check('Example.tfrecord')

    img = np.load('img.npy')
    lab = np.load('lab.npy')

    datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=50,
                                                           width_shift_range=2.0,
                                                           height_shift_range=2.0,
                                                           zoom_range=0.5, fill_mode='nearest',
                                                           horizontal_flip=True, vertical_flip=True,
                                                           rescale=None)

    iterator = datagen.flow(img, lab, batch_size=32, shuffle=True)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        epoch = 100
        for step in range(epoch):
            img_batch_aug, lab_batch_aug = iterator.next()


            img_batch_aug += 1.0
            lab_batch_aug += 1.0


    img, lab, h, w, img_shape = read_tfrecord()

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        epoch = 100

        # to run queue
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(epoch):
            img_batch, lab_batch, h_batch, w_batch = sess.run([img, lab, h, w])
        
            for i in range(5):
                img_each = img_batch[i]
                plt.imshow(img_each)
                plt.show()
                print(img_each.shape)

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()


