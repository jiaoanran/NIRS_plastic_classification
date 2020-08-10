# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:49:55 2020

@author: Annora
"""

import tensorflow as tf
import pandas as pd
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  
import numpy as np
import math

#将数据集分为训练集测试集
def load_data_and_labels():
    '''
    input:siliao excel
    output:X_train(54*1*801), X_test(14*1*801), y_train(54*1*4), y_test arrays
    '''
    oct_df = pd.read_excel("plastic.xlsx")
    oct_df1 = oct_df.iloc[:32,:]
    oct_df2 = oct_df.iloc[32:68,:]
    oct_df3 = oct_df.iloc[68:81,:]
    oct_df4 = oct_df.iloc[81:99,:]
    xdata1,xdata2,xdata3,xdata4 = oct_df1.iloc[:,3:],oct_df2.iloc[:,3:],oct_df3.iloc[:,3:],oct_df4.iloc[:,3:]
    #print(xdata1,xdata2,xdata3,xdata4)
    ydata1 = oct_df1['class']
    ydata2 = oct_df2['class']
    ydata3 = oct_df3['class']
    ydata4 = oct_df4['class']
    X_train1, X_test1, y_train1, y_test1 = train_test_split(xdata1, ydata1,test_size=0.25)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(xdata2, ydata2,test_size=0.25)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(xdata3, ydata3,test_size=0.25)
    X_train4, X_test4, y_train4, y_test4 = train_test_split(xdata4, ydata4,test_size=0.25)
    X_train= pd.concat([X_train1,X_train2,X_train3,X_train4])
    X_test= pd.concat([X_test1,X_test2,X_test3,X_test4])
    y_train= pd.concat([y_train1,y_train2,y_train3,y_train4])
    y_test= pd.concat([y_test1,y_test2,y_test3,y_test4])
    #X_train, X_test = get_sample(X_train,5),get_sample(X_test,5)
    X_train,y_train = shuffle(X_train,y_train)
    X_test,y_test = shuffle(X_test,y_test)
    X_train.to_excel('newXtrain1.xlsx',encoding='utf-8', index=False, header=False)
    X_test.to_excel('newXtest1.xlsx',encoding='utf-8', index=False, header=False)
    y_train.to_excel('newytrain1.xlsx',encoding='utf-8', index=False, header=False)
    y_test.to_excel('newytest1.xlsx',encoding='utf-8', index=False, header=False)
    return

def get_sample(xdata,num):
    x_data = []
    xdata = xdata.values
    for i in range(len(xdata)):#隔5点取样
        xi = xdata[i]
        x_data.append(xi[::num])
    #print(len(x_data[0]))
    x_data = pd.DataFrame(x_data)
    return x_data

def shuffle(xdata,ydata):
    #shuffle
    xdata = xdata.values
    ydata = ydata.values
    index = [i for i in range(len(xdata))]
    np.random.shuffle(index) 
    #print(index)
    xdata = pd.DataFrame(xdata[index])
    ydata = pd.DataFrame(ydata[index])
    return xdata,ydata

load_data_and_labels()