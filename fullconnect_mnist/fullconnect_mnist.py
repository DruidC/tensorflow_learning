# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 00:36:35 2019

@author: 11988
"""

import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 5e-3
epoch = 2000
sum_num = 55000
iteration = 550
batchsize = int(sum_num/iteration)

if batchsize == sum_num/iteration:
    print("parameter is OK!")
else:
    print("parameter is wrong!")
    exit()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

mnist = input_data.read_data_sets('mnist/MNIST_data', one_hot=True)

class forward_network():
    def __init__(self):
        pass
    
    def set_input(self,batchsize):
        x = tf.placeholder(dtype=tf.float32,shape=[None,784],name="input")
        return x
    
    def set_output(self,batchsize):
        y = tf.placeholder(dtype=tf.float32,shape=[None,10],name="input")
        return y
    
    def add_fullconnect(self,input_num,input_size,output_size,activation=None):
        weight = tf.Variable(tf.random_normal(shape=[input_size,output_size],mean=0,stddev=1),name="weight")
        basis = tf.Variable(tf.zeros(shape=[output_size])+0.1,name="basis") 
        output_num = tf.matmul(input_num,weight)+basis
        if activation is None:
            return output_num
        else:
            return activation(output_num)
    
    def set_network(self,input_data,batchsize):
        pre = self.add_fullconnect(input_data,784,10,tf.nn.softmax)
        return pre

class back_network():
    def __init__(self):
        pass
    
    def get_loss(self,truer_data,pre_data):
        loss=tf.reduce_mean(-tf.reduce_sum(truer_data*tf.log(pre_data),reduction_indices=[1]))
        return loss
    
    def trainor(self,learning_rate,loss):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        return train_step
    
    def test_odds(self,truer_data,pre_data):
        correct_prediction = tf.equal(tf.argmax(truer_data, 1), tf.argmax(pre_data, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

forward = forward_network()
x = forward.set_input(batchsize)
y = forward.set_output(batchsize)
pre = forward.set_network(x,batchsize)

back = back_network()
loss = back.get_loss(y,pre)
train = back.trainor(learning_rate,loss)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

saver = tf.train.Saver()
if os.path.exists("ckpt") and os.listdir("ckpt") is not None:
    model_file=tf.train.latest_checkpoint('ckpt/')
    saver.restore(sess,model_file)

for i in range(epoch):
    for ite in range(iteration):
        batch = mnist.train.next_batch(batchsize)
        sess.run(train,feed_dict={x:batch[0],y:batch[1]})
    if i%20 == 0:
        saver.save(sess,'ckpt/checkpoint',global_step=i)
        batch_test = mnist.test.next_batch(batchsize)
        pre_data = sess.run(pre,feed_dict={x:batch_test[0]})
        print("=============epoch:"+str(i)+"=============")
        print("accuracy:"+str(sess.run(back.test_odds(batch_test[1],pre_data))))
        print("loss:"+str(sess.run(loss,feed_dict={x:batch_test[0],y:batch_test[1]})))



