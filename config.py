#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:43:26 2017

@author: pengjy
"""

#定义常量
hidden_units = 1  #(16)     # hidden layer units, the number of units in the LSTM cell
num_input_features = 28     #num of features
output_size = 1
learning_rate = 0.01         #学习率 0.0006
label_index =28
predict_step =2
num_gpu = 2
#num_batch = 10 

batch_size = 32
time_step = 5
train_begin = 0
train_end = 150000
test_begin = 150000
test_end = 180000
num_layers = 1
beta = 0.001

input_filename = 'dscdata1.csv'
save_path = './model/'
logs_path = './logs_path'
max_iters =10


