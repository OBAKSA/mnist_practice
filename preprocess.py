import tensorflow as tf
import numpy as np
import glob
import matplotlib
matplotlib.use('TkAgg')
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


def write_tfrecord(filename):
    """ TODO: write a tfrecord file containing img-lab pairs whose name is filename
        in case you want to save tfrecord files instead of a tfrecord file, change input argument as you want """
    pass


def read_tfrecord(folder):
    """ TODO: read tfrecord files in folder, Return shuffled mini-batch img,lab pairs """
    pass


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


