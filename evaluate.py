#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:22:32 2018

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

import model

def evaluate_images():
    
    data_dir = '/home/caozhang/spyder_projects/mnist_code/mnist_test_images/'
    log_dir = '/home/caozhang/spyder_projects/mnist_code/logs/'
    images_list = []
    
    for file in os.listdir(data_dir):
        images_list.append(data_dir + file)
        
    images = [cv2.imread(images_list[i], cv2.IMREAD_GRAYSCALE) for i in range(len(images_list))]
    image_28x28 =  [cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC) for image in images]
    
    with tf.Graph().as_default() as g:
        for image in image_28x28:
            plt.imshow(image)
            plt.show()
            
            image = tf.cast(image, tf.float32)
            x_imag = tf.reshape(image, [-1, 28, 28, 1])       
            logit = model.inference(x_imag)
            logit = tf.nn.softmax(logit)
            saver = tf.train.Saver()
            
            with tf.Session() as sess:
                # 测试多张图片，我们模型的参数需要重复使用，所以我们需要告诉TF允许复用参数，加上下行代码
                tf.get_variable_scope().reuse_variables()
                print ('Reading checkpoints...')
                ckpt = tf.train.get_checkpoint_state(log_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print ('Loading success! global step is: %s' % global_step)
                else:
                    print ('Not find checkpoint!')
                    return
                
                prediction = sess.run(logit)
                max_index = np.argmax(prediction) 
                if max_index == 0:
                     print ('This is a 0 with possibility %.6f' % prediction[:, 0])
                     
                if max_index == 1:
                     print ('This is a 1 with possibility %.6f' % prediction[:, 1])
                 
                if max_index == 2:
                     print ('This is a 2 with possibility %.6f' % prediction[:, 2])
                     
                if max_index == 3:
                     print ('This is a 3 with possibility %.6f' % prediction[:, 3])
                    
                if max_index == 4:
                     print ('This is a 4 with possibility %.6f' % prediction[:, 4])
                    
                if max_index == 5:
                     print ('This is a 5 with possibility %.6f' % prediction[:, 5])
                     
                if max_index == 6:
                     print ('This is a 6 with possibility %.6f' % prediction[:, 6])
                     
                if max_index == 7:
                     print ('This is a 7 with possibility %.6f' % prediction[:, 7])
                    
                if max_index == 8:
                     print ('This is a 8 with possibility %.6f' % prediction[:, 8])
                    
                if max_index == 9:
                     print ('This is a 9 with possibility %.6f' % prediction[:, 9])