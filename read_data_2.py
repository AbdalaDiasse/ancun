#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:42:36 2017

@author: pengjy
"""

import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import MinMaxScaler 

'''
read data from input_data:filename
'''

def read_raw_data():
    #——————————————————导入数据——————————————————————
    f=open(config.input_filename) 
    df=pd.read_csv(f)     #读入股票数据
    data=df.iloc[:,:].values  #取第3-10列
    return data

def normalized(data):
    sc_X =MinMaxScaler()
    normalized_data = sc_X.fit_transform(data)
    return normalized_data
   
      
def get_train_data():
    time_step = config.time_step
    data = read_raw_data()
    epoch_start_index=[]
    data_train=data[config.train_begin:config.train_end]
    normalized_train_data=normalized(data_train)
    train_x,train_y=[],[]   #训练集 
    for i in range(len(normalized_train_data)-2*time_step):bvn
       if i % config.batch_size==0:
           epoch_start_index.append(i)
       x=normalized_train_data[i:i+time_step,:config.label_index]
       y=normalized_train_data[i+time_step:i+time_step+time_step,config.label_index,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    epoch_start_index.append((len(normalized_train_data)-time_step))
    return epoch_start_index,train_x,train_y

'''
#获取测试集
'''
def get_test_data():
    time_step = config.time_step
    data = read_raw_data()
    data_test=data[config.test_begin:config.test_end]
    normalized_test_data=normalized(data_test)
    test_x,test_y=[],[]  
    for i in range(len(normalized_test_data)-2*time_step):
       x=normalized_test_data[i:i+time_step,:config.label_index]
       y=normalized_test_data[i+time_step+time_step,config.label_index,np.newaxis]
       test_x.append(x.tolist())
       test_y.append(y.tolist())
    return test_x,test_y

if __name__ =='__main__':
    data_train=get_train_data()
    
    data_test=get_test_data()