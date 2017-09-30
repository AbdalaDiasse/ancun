# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:01:24 2017

@author: XQing
"""

''' Multi-GPU Training LSTM.
'''

import numpy as np
import tensorflow as tf
import pandas as pd
import time
import config
import read_data_2
import os



hidden_units =  config.hidden_units      #hidden layer units, the number of units in the LSTM cell
num_input_features = config.num_input_features # Number of features
output_size = config.output_size  # The ouput size of the LSTM
learning_rate = config.learning_rate # Leaning rate of the LSTM
max_iters = config.max_iters  # Maximum of iterations
save_path = config.save_path  # Model savnig Paths
time_step = config.time_step  # LSTM Time Step
batch_size = config.batch_size  #Batch Size
num_layers = config.num_layers  # Number of layers
beta =config.beta #
logs_path =  config.logs_path
num_gpus = config.num_gpu

tf.app.flags.DEFINE_string("type", "", "Either 'train' or 'predict'")
tf.app.flags.DEFINE_integer("index", 5, "Prediction index")
FLAGS = tf.app.flags.FLAGS

epoch_start_index,train_x,train_y = read_data_2.get_train_data()

# Build a convolutional neural network
def lstm_cell(X, reuse):
    # Define a scope for reusing the variables
    with tf.variable_scope('lstm', reuse=reuse):
          
        weights={
             'in':tf.Variable(tf.random_normal([num_input_features,hidden_units])),
             'out':tf.Variable(tf.random_normal([hidden_units,1]))
             }
        biases={
            'in':tf.Variable(tf.constant(0.1,shape=[hidden_units,])),
            'out':tf.Variable(tf.constant(0.1,shape=[1]))
           }

        w_in=weights['in']
        b_in=biases['in']
        batch_size=tf.shape(X)[0]  
        time_step=tf.shape(X)[1]
        X2d = tf.reshape(X,[-1,num_input_features]) 
        inputs =tf.matmul(X2d,w_in)+b_in
        inputs =tf.reshape(inputs,[-1,time_step,hidden_units])

        cell = tf.contrib.rnn.LSTMCell(num_units = hidden_units,\
                                            state_is_tuple = True,\
                                            use_peepholes=True)

        cell = tf.contrib.rnn.MultiRNNCell([cell]*num_layers)
 
        initial_state=cell.zero_state(batch_size,dtype=tf.float32)
        
        outputs,_=tf.nn.dynamic_rnn(cell = cell,\
                              inputs = inputs,\
                              initial_state = initial_state,\
                              dtype=tf.float32) 
        
        outputs_pred = tf.reshape(outputs,[-1,hidden_units])
        w_out=weights['out']
        b_out=biases['out']
        pred = tf.matmul(outputs_pred,w_out)+b_out
    return pred

def total_loss(pred_train,Y,reuse):
      with tf.variable_scope(tf.get_variable_scope()):
            loss_op=tf.reduce_mean(tf.square(tf.reshape(pred_train,[-1])-tf.reshape(Y, [-1])))
            tf.add_to_collection('losses', loss_op)
            _total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            return _total_loss

def tower_loss(scope,X,Y,reuse):
      pred_train =lstm_cell(X,reuse)
      _= total_loss(pred_train,Y,reuse)
      _tower_loss_list = tf.get_collection('losses', scope)
      _tower_loss = tf.add_n(_tower_loss_list, name='tower_loss')
      
      return _tower_loss
          
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            #if g is not None:
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# Place all ops on CPU by default
def training():
      with tf.Graph().as_default(),tf.device('/cpu:0'):
          tower_grads = []
          reuse_vars = False
      
          # tf Graph input
          X=tf.placeholder(tf.float32, shape=[None,time_step,num_input_features],name='X') 
          Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size],name='Y')
          global_step = tf.get_variable(
                      'global_step', [],
                      initializer=tf.constant_initializer(0), trainable=False)
          
          #starter_learning_rate = learning_rate
          #decay_step = len(epoch_start_index)//num_gpus*batch_size 
          #decay_learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
          #                     decay_step, 0.9, staircase=True)
          decay_learning_rate = learning_rate
          optimizer = tf.train.GradientDescentOptimizer(learning_rate=decay_learning_rate)
          # Loop over all GPUs and construct their own computation graph
          for i in range(num_gpus):
              with tf.device('/gpu:%d' % i):
                 with tf.name_scope('Tower_%d'%i) as scope:
                  # Split data between GPUs
                  _x = X[i * batch_size: (i+1) * batch_size]
                  _y = Y[i * batch_size: (i+1) * batch_size]
      
           
                  # Define loss and optimizer (with train logits, for dropout to take effect)
                  #loss_op=tf.reduce_mean(tf.square(tf.reshape(pred_train,[-1])-tf.reshape(_y, [-1])))
                  loss_op = tower_loss(scope,_x,_y,reuse=reuse_vars)
                  # We need to try this :opt = tf.train.GradientDescentOptimizer(lr)
                  
                  # Reuse variables for the next tower.
                  tf.get_variable_scope().reuse_variables()
                  
                  grads = optimizer.compute_gradients(loss_op)
      
                  reuse_vars = True
                  tower_grads.append(grads)

          tower_grads_avg = average_gradients(tower_grads)
          apply_gradient_op = optimizer.apply_gradients(tower_grads_avg,global_step=global_step)
         

          train_op = apply_gradient_op
          pred_test = lstm_cell(X,reuse=True)
          # Initialize the variables (i.e. assign their default value)
          init = tf.global_variables_initializer()
          
         
          total_list = np.zeros(max_iters)
          if os.path.isfile(save_path):
                   os.remove(save_path)    
          saver=tf.train.Saver(tf.global_variables())
          conf=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
          
          # Start Training
          with tf.Session(config = conf) as sess:
              print ('----Training the LSTM With Multi GPU------')
              # Run the initializer
              sess.run(init)
              ts = time.time()
              for i in range(max_iters):
                         #print(i)
                         for epoch in range(len(epoch_start_index)//num_gpus): 
                             _,loss_=sess.run([train_op,loss_op,],feed_dict = \
                                             {X:train_x[epoch_start_index[epoch]:epoch_start_index[epoch]+batch_size*num_gpus],\
                                              Y:train_y[epoch_start_index[epoch]:epoch_start_index[epoch]+batch_size*num_gpus]})
                             total_list[i]= loss_ 
                         if i % 10 == 0:
                                 print(i,loss_)
                         if i % 50==0:
                             print("save model:",saver.save(sess,save_path+'nox.model',global_step=i))
              print("The total Lost=", np.sum(total_list))
              print("Optimization Finished in %s seconds!" %(time.time()-ts))
              
              print ('----Model prediction the LSTM With Multi GPU------')
              test_x,test_y = read_data_2.get_test_data()
              test_predict=[]
              for step in range(len(test_x)):
                prob=sess.run(pred_test,feed_dict={X:[test_x[step]]}) 
                predict=prob.reshape((-1))
                test_predict.append(predict[-1])
              
              test_y=np.array(test_y).reshape(-1,1)
              test_predict=np.array(test_predict).reshape(-1,1)

              mse = ((test_predict-test_y[:len(test_predict)]) ** 2).mean(axis=None)
              print("Mean Square error")
              print (mse)
            
              #Saving the prediction result for further analysis
              df = pd.DataFrame(np.concatenate((test_y ,test_predict), axis=1))
              header =["real_values","predicted_values"]
              df.columns = header
              df.to_csv("pred_result.csv", index_label='Index_name')
             
            
def main(argv=None):     
    print ('----Training the LSTM With Multi GPU------')
    training()  
               
if __name__ =="__main__":
   tf.app.run()              
              
              
   
  