
# coding: utf-8

# In[8]:


# All Includes
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow.python.framework import ops
from sklearn import metrics ,preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, ClassifierMixin

import pandas as pd
import itertools
import os


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing import sequence
from sklearn.metrics import mean_squared_error
from keras.callbacks import Callback
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, AveragePooling1D


# In[19]:




# In[407]:


class RNN_classifier(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, n_cells = 1, n_hidden = 32, n_classes = None,
                 learning_rate = 0.0025,lambda_loss_amount = 0.0015,
                training_iters = 100,  batch_size = None, look_back = 10):
        """
        Called when initializing the classifier
        """
        
        # Input Data 

        self.training_data_count = None  # 7352 training series (with 50% overlap between each serie)
        self.test_data_count = None  # 2947 testing series
        self.n_steps = None
        self.n_input = None  # 9 input parameters per timestep
        self.n_classes = n_classes

        # LSTM Neural Network's internal structure

        self.n_hidden = 32 # Hidden layer num of features
        self.n_classes = 6 # Total classes (should go up, or should go down)
        self.n_cells = n_cells
        self.model = None
        self._sess = None
        
        # To keep track of training's performance
        self.test_losses = []
        self.test_accuracies = []
        self.train_losses = []
        self.train_accuracies = []
        
        # Training 
        self.look_back = look_back
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_iters = training_iters
        self.batch_size = batch_size
        
    def fit(self, X_training, y_training, X_val, y_val, verbose = 0, transform_data = True):


        '''# In[414]:


        model = RNN_classifier(n_classes=6, batch_size=3000,training_iters=10)


        # In[415]:


        df = pd.read_csv("/home/ahmet/notebooks/data/G9_data/processed.csv")
        df.dropna(axis=0, how='any', inplace=True)


        # In[416]:



        # convert an array of values into a dataset matrix
        def create_dataset(X, y, look_back=3):

            dataX, dataY = [], []
            for i in range(look_back , len(X)):
                a = X[i-look_back:i, :]
                b = y[i]
                dataX.append(a)
                dataY.append(b)
            return np.array(dataX), np.array(dataY)


        # In[417]:


        df_training = df[np.logical_and(df['subject_id']!=15,df['subject_id']!=14 )]
        df_test = df[np.logical_or(df['subject_id']==15, df['subject_id']==14 )]


        # In[418]:


        X_test_used = df_test[df_test.columns[:-3]].values
        y_test = df_test['label'].values -1
        X_train_used = df_training[df_training.columns[:-3]].values
        y_train = df_training['label'].values -1


        # In[419]:


        X_test_used, y_test = create_dataset(X_test_used, y_test, look_back=10)
        X_train_used, y_train = create_dataset(X_train_used, y_train, look_back=10)


        # In[424]:


        model.fit(X_training=X_train_used, y_training= y_train, X_val=X_test_used, y_val=y_test, verbose=1)


        # In[425]:


        a = model.predict(X_test_used, y_test)


        # In[426]:


        metrics.accuracy_score( y_test,a)


        '''
        # Prepare dataset 
        if transform_data:
            X_val, y_val = self.transform_dataset(X_val, y_val)
            X_training, y_training = self.transform_dataset(X_training, y_training)
        
        # Set parameters from data 
        self.training_data_count = len(X_training)
        self.test_data_count = len(X_val)
        self.n_steps = len(X_training[0])  
        self.n_input = len(X_training[0][0])
        ops.reset_default_graph()
        
        # Graph input/output
        self._x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_input])
        self._y = tf.placeholder(tf.float32, [None, self.n_classes])
        # Graph weights
        self._weights = {
            'hidden': tf.Variable(tf.random_normal([self.n_input, self.n_hidden])), # Hidden layer weights
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes], mean=1.0))
        }
        self._biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        self.model = self._LSTM_RNN(self._x, self._weights, self._biases)

        # Loss, optimizer and evaluation
        l2 = self.lambda_loss_amount * sum(
            tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
        ) # L2 loss prevents this overkill neural network to overfit the data
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=self.model)) + l2 # Softmax loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost) # Adam Optimizer

        self.correct_pred = tf.equal(tf.argmax(self.model,1), tf.argmax(self._y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        
        
        # To keep track of training's performance
        self.test_losses = []
        self.test_accuracies = []
        self.train_losses = []
        self.train_accuracies = []
        saver = tf.train.Saver()
        
        
        # Launch the graph
        self._sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        init = tf.global_variables_initializer()
        self._sess.run(init)
        


        # Perform Training steps with "batch_size" amount of example data at each loop
        step = 1
        epoch = 0
        while step * self.batch_size <= self.training_iters * self.training_data_count:
            batch_xs =         self.extract_batch_size(X_training, step)
            batch_ys =         self.extract_batch_size(y_training, step)
            batch_xs, batch_ys = shuffle(batch_xs, batch_ys, random_state=44)
            # Fit training using batch data
            _, loss, acc = self._sess.run(
                [optimizer, cost, self.accuracy],
                feed_dict={
                    self._x: batch_xs, 
                    self._y: self.one_hot(batch_ys)
                }
            )
            self.train_losses.append(loss)
            self.train_accuracies.append(acc)
            
            if verbose ==1 :
                # Evaluate network only at some steps for faster training: 
                if (step % (self.training_data_count //self.batch_size)) ==0:
                    epoch += 1
                    # To not spam console, show training accuracy/loss in this "if"
                    print("Epoch #" + str(epoch) +                           ":   Batch Loss = " + "{:.6f}".format(loss) +                           ", Accuracy = {}".format(acc))

                    # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
                    loss, acc = self._sess.run(
                        [cost, self.accuracy], 
                        feed_dict={
                            self._x: X_val,
                            self._y: self.one_hot(y_val)
                        }
                    )
                    self.test_losses.append(loss)
                    self.test_accuracies.append(acc)
                    print("PERFORMANCE ON TEST SET: " +                           "Batch Loss = {}".format(loss) +                           ", Accuracy = {}".format(acc))

            step += 1

        print("Optimization Finished!")

        # Accuracy for test data

        one_hot_predictions, accuracy, final_loss = self._sess.run(
            [self.model, self.accuracy, cost],
            feed_dict={
                self._x: X_val,
                self._y: self.one_hot(y_val)
            }
        )

        self.test_losses.append(final_loss)
        self.test_accuracies.append(accuracy)

        return self

    def predict(self, X, y=None, transform_data = True):
        one_hot_predictions = self.predict_proba(X,y, transform_data)


        return(np.argmax(one_hot_predictions, 1))

    def predict_proba(self, X, y=None, transform_data = True):
        try:
            getattr(self, "model")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        if transform_data:
            X,y = self.transform_dataset(X,np.ones(X.shape[0]))
        else:
            y = np.ones(X.shape[0])
        one_hot_predictions = self._sess.run(
            [self.model],
            feed_dict={
                self._x: X,
                self._y:self.one_hot(y)
            }
        )
        return np.array(one_hot_predictions).reshape(-1,self.n_classes)
    def _LSTM_RNN(self, _X, _weights, _biases):
        # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters. 
        # Moreover, two LSTM cells are stacked which adds deepness to the neural network. 
        # Note, some code of this notebook is inspired from an slightly different 
        # RNN architecture used on another dataset, some of the credits goes to 
        # "aymericdamien" under the MIT license.

        # (NOTE: This step could be greatly optimised by shaping the dataset once
        # input shape: (batch_size, n_steps, n_input)
        _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [-1, self.n_input]) 
        # new shape: (n_steps*batch_size, n_input)

        # Linear activation
        _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(_X, self.n_steps, 0) 
        # new shape: n_steps * (batch_size, n_hidden)

        # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        cells = [tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True) for _ in range(self.n_cells)]
        lstm_cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        # Get LSTM cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

        # Get last time step's output feature for a "many to one" style classifier, 
        # as in the image describing RNNs at the top of this page
        lstm_last_output = outputs[-1]

        # Linear activation
        return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']
    
    def extract_batch_size(self,_train, step):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
        shape = list(_train.shape)
        shape[0] = self.batch_size
        batch_s = np.empty(shape)

        for i in range(self.batch_size):
            # Loop index
            index = ((step-1)*self.batch_size + i) % len(_train)
            batch_s[i] = _train[index] 

        return batch_s
    def one_hot(self, y_):
        # Function to encode output labels from number indexes 
        # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

        y_ = y_.reshape(len(y_))
        return np.eye(self.n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

    
    # convert an array of values into a dataset matrix
    def transform_dataset(self,X, y, look_back=3):

        dataX, dataY = [], []
        for i in range(look_back , len(X)):
            a = X[i-look_back:i, :]
            b = y[i]
            dataX.append(a)
            dataY.append(b)
        return np.array(dataX), np.array(dataY)
    

    
    
    
class KerasClassifier(BaseEstimator, ClassifierMixin): 
    def __init__(self, batch_size, n_classes, model = 'MLP', n_epoch = 50):
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_epoch = n_epoch
        self.timesteps = None
        if model =='MLP':
            self.model = self._MLP()
        elif model =='CNN':
            self.model = self._CNN()

            
            
    def fit(self, X_training, y_training, X_val, y_val):
        self.timesteps = len(X_training[0])
        self.model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
        history =model.fit(X_training,
                  self.one_hot(y_training),
                  batch_size=self.batch_size,
                  validation_data=(X_val, self.one_hot(y_val)),
                  epochs=self.n_epoch)
        return history
        
    def one_hot(self, y_):
        # Function to encode output labels from number indexes 
        # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

        y_ = y_.reshape(len(y_))
        return np.eye(self.n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS
    
    def predict(self, X, y=None):
        one_hot_predictions = self.predict_proba(X)
        return(np.argmax(one_hot_predictions, 1))

    def predict_proba(self, X):
        
        return self.model.predict_proba(X)
    
    def _MLP(self):
        print('Build model...')

        model = Sequential()
        model.add(Dense(512, input_shape =(self.timesteps,)))
        model.add(Dense(512,activation='tanh'))
        model.add(Dense(512, activation='tanh'))
        model.add(Dense(512, activation='tanh'))
        model.add(Dense(256))
        model.add(Dense(256))
        model.add(Dense(32))
        model.add(Dense(self.n_classes, activation='sigmoid'))
        return model
    def _CNN(self):
        print('Build model...')
        model = Sequential()
        model.add(Conv1D(32,
                        16,
                         input_shape=(self.timesteps,1),
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(MaxPooling1D(6))
        model.add(Conv1D(64,
                        4,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(MaxPooling1D(6))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(32))
        model.add(Dense(self.n_classes, activation='softmax'))

        return model

