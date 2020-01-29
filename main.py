import numpy as np
import glob
import os
import tensorflow as tf
from preprocess import *
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('process', 'write', 'which process you want to do: write(write tfrecord file), train(train model), test(test model)')
tf.flags.DEFINE_string('gpu', '0', 'gpu number to be used')


def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    if FLAGS.process == 'write':
        write_tfrecord('FILE-NAME.tfrecord')
    elif FLAGS.process == 'train':



if __name__ == "__main__":
    tf.app.run()