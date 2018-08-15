# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import LeNet

# 参数
BATCH_SIZE = 128
EPOCHS = 10
RATE = 0.1

def train(mnist):
    x_train, y_train = mnist.train.images, mnist.train.labels
    x_validation, y_validation = mnist.validation.images, mnist.validation.labels
    # print(x_train[0].shape)
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)),'constant')
    x_validation = np.pad(x_validation, ((0, 0), (2, 2), (2, 2), (0, 0)),'constant')
    # print(x_train[0].shape)

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
    y = tf.placeholder(tf.int32, shape=[None, ])
    one_hot_y = tf.one_hot(y, 10)

    y_ = LeNet.LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(RATE).minimize(cross_entropy_mean)

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(one_hot_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for i in range(EPOCHS):
            x_train, y_train = shuffle(x_train, y_train)
            for offset in range(0, len(x_train),BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                session.run(train_step, feed_dict={x:batch_x, y:batch_y})
            print("EPOCHS:", i+1)
            accuracy_score = session.run(accuracy, feed_dict={x:x_validation, y:y_validation})
            print('Validation Accuracy', accuracy_score)
        # test
        x_test, y_test = mnist.test.images, mnist.test.labels
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)),'constant')
        test_accuracy = session.run(accuracy, feed_dict={x:x_test, y:y_test})
        print('Test Accuracy', test_accuracy) # test_accuracy = 0.9876


def main(argv=None):
    mnist =input_data.read_data_sets("MNIST_data/", reshape=False)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
