# -*- coding: utf-8 -*-
"""
Created on Sat May  9 07:44:24 2020

@author: Annora
"""
import tensorflow as tf
import pandas as pd
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  
import numpy as np
import math


#================参数=======================
SEQUENCE_LENGTH = 1501
OUTPUT_LENGTH = 4
CONV_SIZE1 = 2 # 卷积核宽度方向的大小
CONV_SIZE2 = 2
CONV_NUM = 3#卷积核个数
CONV_NUM2 = 12
NUM3 = 60#全连接层
CONV_STRIDE = 1 #卷积核在宽度上步长
POOL_SIZE = 2 #池化层核宽度方向上的大小
POOL_STRIDE = 2  # 卷积核宽度方向上的步长
LEARNING_RATE = 1e-4
EPOCH = 5501
lamda = 0.5
#===========================================
def load_data():
    # X_train = pd.read_excel('newXtest.xlsx')
    # y_train = pd.read_excel('newytest.xlsx')
    X_train = pd.read_excel('newXtest.xlsx')
    y_train = pd.read_excel('newytest.xlsx')
    X_test = X_train.values
    y_test = y_train.values
    
    #X_train, X_validation, y_train, y_validation = train_test_split(xdata, ydata,test_size=1/10)
    return X_test, y_test


def weight_variable(shape): 
    initial=tf.truncated_normal(shape,stddev=0.1)
    W = tf.Variable(initial)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    return tf.Variable(initial)

def bias_variable(shape): 
    initial=tf.constant(0.1,shape=shape) 
    return tf.Variable(initial)

def conv2d(x,W):#卷积层
    #stride[1,x_movement,y_movement,1]
	return tf.nn.conv2d(x,W,strides=[1,CONV_STRIDE,1,1],padding='SAME') 

def max_pool_1xn(x): 
	return tf.nn.max_pool(x,ksize=[1,POOL_SIZE,1,1],strides=[1,POOL_STRIDE,1,1],padding='SAME')
    
def oneDCNN():
    ops.reset_default_graph()#重新运行模型而不覆盖tf变量
    sess = tf.InteractiveSession()
    #define placeholder for inputs to network
    xs = tf.placeholder(tf.float32,[None,SEQUENCE_LENGTH])
    ys = tf.placeholder(tf.float32,[None,OUTPUT_LENGTH])
    keep_prob = tf.placeholder(tf.float32)
    x_sequence=tf.reshape(xs,[-1,SEQUENCE_LENGTH,1,1])
    
    ##conv1 layer##
    W_conv1=weight_variable([CONV_SIZE1,1,1,CONV_NUM])
    b_conv1=bias_variable([CONV_NUM])
    h_conv1=tf.nn.leaky_relu(conv2d(x_sequence,W_conv1)+b_conv1)
    h_pool1=max_pool_1xn(h_conv1)
    
    ##func1 layer##
    W_fc1 = weight_variable([int(SEQUENCE_LENGTH/(POOL_SIZE)+1) *1* CONV_NUM, NUM3])
    b_fc1 = bias_variable([NUM3])
    h_pool1_flat=tf.reshape(h_pool1,[-1,int(SEQUENCE_LENGTH/(POOL_SIZE)+1) *1* CONV_NUM]) 
    h_fc1=tf.nn.leaky_relu(tf.matmul(h_pool1_flat,W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    ##func2 layer##
    W_fc2 = weight_variable([NUM3, OUTPUT_LENGTH])
    b_fc2 = bias_variable([OUTPUT_LENGTH])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    #the error
    regularizer = tf.contrib.layers.l2_regularizer(lamda)
    reg_term = tf.contrib.layers.apply_regularization(regularizer)
    cross_entropy = -tf.reduce_sum(ys*tf.log(y_conv))+reg_term
    #cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys-y_conv),reduction_indices = [1]))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    #train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
    #train_step = tf.train.MomentumOptimizer(LEARNING_RATE,M).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

     #加载
    X_test, y_test = load_data()
    ytest = tf.one_hot([i[0] for i in y_test],4)

    saver = tf.train.Saver()
    with tf.Session() as sess:  
        ytest = sess.run(ytest)
        saver.restore(sess,'model/save_net.ckpt')
        weights=sess.run(W_fc2)
        bias = sess.run(b_fc2)
        #print("weights：",weights)
        #print("bias：",bias)
        #loss
        acc = sess.run(accuracy, feed_dict={xs:X_test,ys:ytest,keep_prob:1})
        print(acc)
        
    return 

if __name__ == '__main__':
    oneDCNN()
        