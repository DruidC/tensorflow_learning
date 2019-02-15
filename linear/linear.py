# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 18:53:59 2019

@author: 11988
"""

import tensorflow as tf
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

learning_rate = 0.0001
epoch = 100
sum_num = 300
iteration = 6
batchsize = int(sum_num/iteration)

if batchsize == sum_num/iteration:
    print("parameter is OK!")
else:
    print("parameter is wrong!")
    exit()

class forward_network():
    def __init__(self):
        pass
    
    def set_input(self,batchsize):
        x = tf.placeholder(dtype=tf.float32,shape=[batchsize,1],name="input")
        return x
    
    def set_output(self,batchsize):
        y = tf.placeholder(dtype=tf.float32,shape=[batchsize,1],name="output")
        return y
    
    def add_fullconnect(self,input_num,input_size,output_size,activation=None):
        weight = tf.Variable(tf.random_normal(shape=[input_size,output_size],mean=0,stddev=1),name="weight")
        basis = tf.Variable(tf.zeros(shape=[batchsize,output_size])+0.1,name="basis") 
        output_num = tf.matmul(input_num,weight)+basis
        if activation is None:
            return output_num
        else:
            return activation(output_num)
    
    def form_network(self,x):
        l1 = self.add_fullconnect(x,1,100,activation=tf.nn.relu)
        pre = self.add_fullconnect(l1,100,1,activation=tf.nn.relu)
        return pre

class back_network():
    def __init__(self):
        pass
    
    def get_loss(self,truer_num,pre_num):
        loss = tf.reduce_mean(tf.square(truer_num-pre_num))
        return loss
    
    def trainor(self,loss,learning_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(loss)
        return train

x_data = np.linspace(0,1,num=300).reshape([300,1])
noise = np.random.random_sample(size=x_data.shape)/20
y_data = np.square(x_data)+noise

forward = forward_network()
x = forward.set_input(batchsize)
y = forward.set_output(batchsize)
pre = forward.form_network(x)

back = back_network()
loss = back.get_loss(truer_num=y,pre_num=pre)
train = back.trainor(loss=loss,learning_rate=learning_rate)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

saver = tf.train.Saver()
if os.path.exists("ckpt") and os.listdir("ckpt") is not None:
    model_file=tf.train.latest_checkpoint('ckpt/')
    saver.restore(sess,model_file)

for i in range(epoch):
    for ite in range(iteration):
        sess.run(train,feed_dict={x:x_data[ite*batchsize:ite*batchsize+batchsize],y:y_data[ite*batchsize:ite*batchsize+batchsize]})
    if i%10 == 0:
        saver.save(sess,'ckpt/checkpoint',global_step=i)
    print("===============epoch:"+str(i+1)+"===============")
    print(sess.run(loss,feed_dict={x:x_data[0:300:6],y:y_data[0:300:6]}))
    



