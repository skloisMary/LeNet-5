# coding=utf-8
# 由于minst的图像输入是28*28，而LeNet要求的输入大小为32*32，所以经常用一个类似
# LeNet-5模型的卷积神经网络来解决MNIST数字识别问题
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# 输入节点和输出结点大小
inputNode = 784
outputNode = 10


def Like_LeNet(input_tensor):
    input_tensor = tf.reshape(input_tensor, [-1, 28, 28, 1])
    # conv1 filter 5*5*32 input=28*28*1 output= 28*28*32
    conv1_w = tf.Variable(tf.truncated_normal([5, 5, 1, 32], mean=0, stddev=0.1))
    conv1_b =tf.Variable(tf.zeros(32))
    conv1 = tf.nn.conv2d(input_tensor, conv1_w, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    conv1 = tf.nn.relu(conv1)
    # print(conv1.shape)

    # pooling input=28*28*32 output=14*14*32
    pooling_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # print(pooling_1.shape)


    # conv2 filter 5*5*64 input=14*14*32 output=14*14*64
    conv2_w = tf.Variable(tf.truncated_normal([5, 5, 32, 64], mean=0, stddev=0.1))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2 = tf.nn.conv2d(pooling_1, conv2_w, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
    conv2 = tf.nn.relu(conv2)
    # print(conv2.shape)

    # pooling input=14*14*64 output=7*7*64
    pooling_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # print(pooling_2.shape)


    # FC1 input=7*7*64  output=7*7*64=3136
    fc1 = flatten(pooling_2)
    # print(fc1.shape)

    # input=3136, output= 512
    fc1_w = tf.Variable(tf.truncated_normal(shape=(3136, 512), mean=0, stddev=0.1))
    fc1_b = tf.Variable(tf.zeros(512))
    fc1 = tf.nn.relu(tf.matmul(fc1, fc1_w) + fc1_b)
    # print(fc1.shape)

    # FC2 input=512, output =10
    fc2_w = tf.Variable(tf.truncated_normal(shape=(512, 10), mean=0, stddev=0.1))
    fc2_b = tf.Variable(tf.zeros(10))
    logits = tf.nn.relu(tf.matmul(fc1, fc2_w)) + fc2_b
    return logits






