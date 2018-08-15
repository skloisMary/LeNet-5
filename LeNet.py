# coding=utf-8
import tensorflow as tf
from tensorflow.contrib.layers import flatten


# 定义网络
def LeNet(input_tensor):
    # C1  conv  Input=32*32*1, Output=28*28*6
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=0, stddev=0.1))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(input_tensor, conv1_w, strides=[1, 1, 1, 1], padding='VALID')+conv1_b
    conv1 = tf.nn.relu(conv1)

    # S2 Pooling Input=28*28*6 Output=14*14*6
    pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # C3 conv Input=14*14*6 Output=10*10*6
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=0, stddev=0.1))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID')+conv2_b
    conv2 = tf.nn.relu(conv2)

    # S4 Pooling Input=10*10*6 OutPut=5*5*16
    pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten Input=5*5*16 Output=400
    fc1 = flatten(pool_2)

    # C5 conv Input=5*5*16=400 Output=120
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=0, stddev=0.1))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b

    # F6 Input=120 OutPut=84
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=0, stddev=0.1))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w)+fc2_b
    fc2 = tf.nn.relu(fc2)

    # F7 Input=84  Output=10
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=0, stddev=0.1))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits

