#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:44:22 2018

@author: caozhang
"""

from  __future__ import absolute_import
from  __future__ import division
from  __future__ import print_function

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

N_CLASSES = 10
LEARNING_RATE = 0.001
MAX_STEP = 30000

def inference(images):
    """
    Args:
        images: input data
    Returns:
        logits
    """
    # conv1
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5, 5, 1, 32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1, dtype=tf.float32))
        conv = tf.nn.conv2d(images, weights, strides = [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        
    # pool
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], 
                               strides=[1, 2, 2, 1], padding='SAME', name=scope.name)
        
    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5, 5, 32, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1, dtype=tf.float32))
        conv = tf.nn.conv2d(pool1, weights, strides = [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        
    # pool2
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name=scope.name)
        
    # local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[-1, 7*7*64])  # flatten
        
        weights = tf.get_variable('weights',
                                  shape=[7*7*64, 1024],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[1024],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1, dtype=tf.float32))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        
#    # dropout
#    with tf.name_scope('dropout') as scope:
#        keep_prob = tf.placeholder(tf.float32)
#        local3_drop = tf.nn.dropout(local3, keep_prob, name=scope.name)
        
    # softmax_linear
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights',
                                  shape=[1024, N_CLASSES],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[N_CLASSES],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1, dtype=tf.float32))
        softmax_linear = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        
    return softmax_linear        

def train():
    train_dir = '/home/caozhang/spyder_projects/mnist_code/mnist_data'
    log_dir = '/home/caozhang/spyder_projects/mnist_code/logs'
    tensorboard_dir = '/home/caozhang/spyder_projects/mnist_code/tensorboard_file'
    # 使标签数据是"one-hot vectors"。 一个one_hot向量除了某一位的数字是1以外其余各维度数字都是0
    mnist = input_data.read_data_sets(train_dir, one_hot=True)
    
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x_input')
    labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels_input')
    
    x_image = tf.reshape(x, shape=[-1, 28, 28, 1])   
    logits = inference(x_image)
    
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels,
                                                                name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss)
        
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        correct_pred = tf.cast(correct_pred, tf.float32)
        accuracy = tf.reduce_mean(correct_pred)
        tf.summary.scalar('accuracy', accuracy)
        
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(loss)
        
    # model saver and tensorboard event saver
    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
        try:
            for step in range(MAX_STEP):
                batch = mnist.train.next_batch(100)
                sess.run(train_op, feed_dict={x: batch[0], labels: batch[1]})
                loss_value = sess.run(loss, feed_dict={x: batch[0], labels: batch[1]})
                accuracy_value = sess.run(accuracy, feed_dict={x: batch[0], labels: batch[1]})
                
                if step % 50 == 0:
                    print ('step: %d, loss: %f, accuracy: %f' % (step, loss_value, accuracy_value))
                    summary_str = sess.run(summary_op, feed_dict={x: batch[0], labels: batch[1]})
                    summary_writer.add_summary(summary_str, step)
                    
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
    
        except tf.errors.OutOfRangeError:
            print ('Done training!')
        
        