import tensorflow as tf
from preprocess import read_tfrecord

# weight and bais wrappers
def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)

def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initial)

def conv_layer(x, filter_size, num_filters, stride, name='conv'):
    """
    Create a 2D convolution layer
    :param x: input from previous layer
    :param filter_size: size of each filter
    :param num_filters: number of filters (or output feature maps)
    :param stride: filter stride
    :param name: layer name
    :return: The output array
    """
    with tf.variable_scope(name):
        num_in_channel = x.get_shape().as_list()[-1]
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        W = weight_variable(shape=shape)
        tf.summary.histogram('weight', W)
        b = bias_variable(shape=[num_filters])
        tf.summary.histogram('bias', b)
        layer = tf.nn.conv2d(x, W,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        layer += b
        return tf.nn.relu(layer)

def max_pool(x, ksize, stride, name='pool'):
    """
    Create a max pooling layer
    :param x: input to max-pooling layer
    :param ksize: size of the max-pooling filter
    :param stride: stride of the max-pooling filter
    :param name: layer name
    :return: The output array
    """
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding="SAME",
                          name=name)

def flatten_layer(layer):
    """
    Flattens the output of the convolutional layer to be fed into fully-connected layer
    :param layer: input array
    :return: flattened array
    """
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat


def fc_layer(x, num_units, name='fc', use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    with tf.variable_scope(name):
        in_dim = x.get_shape()[1]
        W = weight_variable(shape=[in_dim, num_units])
        tf.summary.histogram('weight', W)
        b = bias_variable(shape=[num_units])
        tf.summary.histogram('bias', b)
        layer = tf.matmul(x, W)
        layer += b
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer


class Classifier(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def build(self, x, reuse=None):
        """ TODO: define your model (2 conv layers and 2 fc layers?)
        x: input image
        logit: network output w/o softmax """
        with tf.variable_scope('model', reuse=reuse):
            x = conv_layer(x, 3, 16, 1, 'conv1')
            x = max_pool(x, 2, 2, 'pool1')
            x = conv_layer(x, 3, 32, 1, 'conv2')
            x = max_pool(x, 2, 2, 'pool2')
            x = flatten_layer(x)
            x = fc_layer(x, 100, 'fc1', use_relu=True)
            x = fc_layer(x, 10, 'fc2', use_relu=False)
            logit = x

        return logit

    def accuracy(self, label_onehot, logit):
        softmax = tf.nn.softmax(logit, -1)
        prediction = tf.argmax(softmax, -1)
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label_onehot, -1), prediction), tf.float32))

    def train(self):
        """ train 10-class MNIST classifier """

        # load data
        tr_img, tr_lab = read_tfrecord(self.FLAGS.datadir, self.FLAGS.batch, self.FLAGS.epoch)
        val_img, val_lab = read_tfrecord(self.FLAGS.val_datadir, self.FLAGS.batch, self.FLAGS.epoch)

        # graph
        tr_logit = self.build(tr_img)
        val_logit = self.build(val_img, True)

        step = tf.Variable(0, trainable=False)
        increment_step = tf.assign_add(step, 1)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tr_lab, logits=tr_logit))
        optimizer = tf.train.AdamOptimizer(self.FLAGS.lr).minimize(loss, global_step=step)

        tr_accuracy = self.accuracy(tr_lab, tr_logit)
        val_accuracy = self.accuracy(val_lab, val_logit)

        saver = tf.train.Saver(max_to_keep=2)
        # session
        with tf.Session() as sess:
            if self.FLAGS.restore:
                saver.restore(sess, tf.train.latest_checkpoint(self.FLAGS.ckptdir))
            else:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                min_val_acc = 10000.
                while True:
                    global_step = sess.run(step)
                    print(global_step)
                    batch_loss, batch_acc, _ = sess.run([loss, tr_accuracy, optimizer])
                    if global_step%10 == 0:
                        print('step:: %d, loss= %.3f, accuracy= %.3f' % (global_step, batch_loss, batch_acc))

                    if global_step%30 == 0:
                        val_acc = sess.run(val_accuracy)
                        print('val accuracy= %.3f' % val_acc)
                        if val_acc < min_val_acc:
                            min_val_acc = val_acc
                            save_path = saver.save(sess, self.FLAGS.ckptdir+'/model_%.3f.ckpt' % val_acc, global_step = step)
                            print('model saved in file: %s' % save_path)

                    sess.run(increment_step)

            except KeyboardInterrupt:
                print('keyboard interrupted')
                coord.request_stop()
            except Exception as e:
                coord.request_stop(e)
            finally:
                save_path = saver.save(sess, self.FLAGS.ckptdir+'/model.ckpt', global_step = step)
                print('model saved in file : %s' % save_path)
                coord.request_stop()
                coord.join(threads)

    def test(self):
        # load data
        ts_img, ts_lab = read_tfrecord(self.FLAGS.datadir, self.FLAGS.batch, 1)

        # graph
        ts_logit = self.build(ts_img)

        ts_accuracy = self.accuracy(ts_lab, ts_logit)

        saver = tf.train.Saver()

        # session
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            saver.restore(sess, tf.train.latest_checkpoint(self.FLAGS.ckptdir))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            total_acc = 0.
            steps = 0
            while True:
                batch_acc = sess.run(ts_accuracy)
                total_acc += batch_acc
                steps += 1
                if steps%10==0:
                    print(steps)

            total_acc /= steps
            print('number: %d, total acc: .1%f%' % (steps, total_acc*100))

            coord.request_stop()
            coord.join(threads)

