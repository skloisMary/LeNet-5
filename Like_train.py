# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import Like_LeNet

# 参数
RATE = 0.0001
EPOCH = 20
BATCH_SIZE = 128

def train(mnist):
   x = tf.placeholder(tf.float32, [None, 784])
   y = tf.placeholder(tf.float32, [None, 10])

   y_ = Like_LeNet.Like_LeNet(x)
   cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y)
   cross_entropy_mean = tf.reduce_mean(cross_entropy)
   # train_step = tf.train.GradientDescentOptimizer(RATE).minimize(cross_entropy_mean)
   train_step = tf.train.AdamOptimizer(RATE).minimize(cross_entropy_mean) # test accuracy=0.9888 RATE=0.0001

   correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

   with tf.Session() as session:
       session.run(tf.global_variables_initializer())
       x_train, y_train = mnist.train.images, mnist.train.labels
       for i in range(EPOCH):
           x_train, y_train = shuffle(x_train, y_train)
           print('EPOCH:', i)
           for offset in range(0, len(x_train), BATCH_SIZE):
               batch_x, batch_y = x_train[offset: offset+BATCH_SIZE], y_train[offset:offset+BATCH_SIZE]
               session.run(train_step, feed_dict={x:batch_x, y:batch_y})
           validation_accuracy = session.run(accuracy,
                                             feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
           print('valiation accuracy:', validation_accuracy)
       # test
       test_accuracy = session.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
       print('test accuracy:', test_accuracy) #


def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()



