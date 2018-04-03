
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import pandas as pd
import itertools
import os
from sklearn.metrics import confusion_matrix

import seaborn as sns
import datetime
from time import sleep
import scipy.signal as signal
from detect_peaks import detect_peaks
from collections import Counter
sns.set(style="darkgrid")


# In[3]:


# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    'AccX',
    'AccY',
    'AccZ',
    'GyroX',
    'GyroY',
    'GyroZ'
]

# Output classes to learn how to classify
LABELS = ['walking',
        'walking upstairs',
        'walking downstairs',
        'sitting',
        'standing',
        'laying']
RAW_DATA_PATH = "G9_data/sippets_eq_sized/"
SUBJECT_LIST = [  1,   3,   4,   5,   6,   7,   8,   9,  10,  11,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
        25,  26]


# In[4]:

def LSTM_fit_predict(test_subjects):
    def load_data(Snippet_paths, n = 128):
        X_signals = []
        y_signals = []
        usecols = INPUT_SIGNAL_TYPES + ['label']
        for snippet_path in Snippet_paths:
            df = pd.read_csv(snippet_path, usecols=usecols)
            y_signals.append(np.array(df['label'][0].reshape(1), dtype=np.int32))
            df = df[INPUT_SIGNAL_TYPES]
            c = df.values.reshape(n, df.shape[1])
            X_signals.append(c)

        return np.asarray(X_signals), np.asarray(y_signals)




    # In[5]:


    n = 128
    Snippet_paths_train =[]
    Snippet_paths_test =[]
    for fol in LABELS:
        folder = RAW_DATA_PATH+str(n)+'/'+fol+'/'
        for root,dirs,files in os.walk(folder):
            for file_ in files:
                if file_.endswith(".csv"):
                    if not int(file_.split("_")[0]) in [test_subjects]:
                        Snippet_paths_train.append(folder+file_)
                    elif int(file_.split("_")[0]) in [test_subjects]:
                        Snippet_paths_test.append(folder+file_)


    # In[6]:


    X_train,y_train = load_data(Snippet_paths_train,128)


    # In[7]:


    X_test,y_test = load_data(Snippet_paths_test,128)


    # In[8]:


    y_test = y_test - 1
    y_train = y_train - 1


    # In[9]:


    #X_train.shape, y_train.shape


    # ## Additionnal Parameters:
    #
    # Here are some core parameter definitions for the training.
    #
    # The whole neural network's structure could be summarised by enumerating those parameters and the fact an LSTM is used.

    # In[10]:


    # Input Data

    training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
    test_data_count = len(X_test)  # 2947 testing series
    n_steps = len(X_train[0])  # 128 timesteps per series
    n_input = len(X_train[0][0])  # 9 input parameters per timestep


    # LSTM Neural Network's internal structure

    n_hidden = 32 # Hidden layer num of features
    n_classes = 6 # Total classes (should go up, or should go down)


    # Training

    learning_rate = 0.0025
    lambda_loss_amount = 0.0015
    training_iters = training_data_count * 300  # Loop 300 times on the dataset
    batch_size = 1500
    display_iter = 30000  # To show test set accuracy during training


    # Some debugging info

    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("(X shape, y shape, every X's mean, every X's standard deviation)")
    print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")


    # ## Utility functions for training:
    #

    # In[11]:


    def LSTM_RNN(_X, _weights, _biases):
        # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
        # Moreover, two LSTM cells are stacked which adds deepness to the neural network.
        # Note, some code of this notebook is inspired from an slightly different
        # RNN architecture used on another dataset, some of the credits goes to
        # "aymericdamien" under the MIT license.

        # (NOTE: This step could be greatly optimised by shaping the dataset once
        # input shape: (batch_size, n_steps, n_input)
        _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [-1, n_input])
        # new shape: (n_steps*batch_size, n_input)

        # Linear activation
        _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(_X, n_steps, 0)
        # new shape: n_steps * (batch_size, n_hidden)

        # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        # Get LSTM cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

        # Get last time step's output feature for a "many to one" style classifier,
        # as in the image describing RNNs at the top of this page
        lstm_last_output = outputs[-1]

        # Linear activation
        return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


    def extract_batch_size(_train, step, batch_size):
        # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

        shape = list(_train.shape)
        shape[0] = batch_size
        batch_s = np.empty(shape)

        for i in range(batch_size):
            # Loop index
            index = ((step-1)*batch_size + i) % len(_train)
            batch_s[i] = _train[index]

        return batch_s


    def one_hot(y_):
        # Function to encode output labels from number indexes
        # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

        y_ = y_.reshape(len(y_))
        n_values = 6
        return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


    # ## Let's get serious and build the neural network:
    #

    # In[12]:


    from tensorflow.python.framework import ops
    ops.reset_default_graph()
    sess = tf.InteractiveSession()


    # In[13]:


    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = LSTM_RNN(x, weights, biases)

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    ) # L2 loss prevents this overkill neural network to overfit the data
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    # ## Hooray, now train the neural network:
    #

    # In[ ]:


    # To keep track of training's performance
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []

    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Perform Training steps with "batch_size" amount of example data at each loop
    step = 1
    while step * batch_size <= training_iters:
        batch_xs =         extract_batch_size(X_train, step, batch_size)
        batch_ys = one_hot(extract_batch_size(y_train, step, batch_size))

        # Fit training using batch data
        _, loss, acc = sess.run(
            [optimizer, cost, accuracy],
            feed_dict={
                x: batch_xs,
                y: batch_ys
            }
        )
        train_losses.append(loss)
        train_accuracies.append(acc)

        # Evaluate network only at some steps for faster training:
        if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):

            # To not spam console, show training accuracy/loss in this "if"
            #print("Training iter #" + str(step*batch_size) +               ":   Batch Loss = " + "{:.6f}".format(loss) +               ", Accuracy = {}".format(acc))

            # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            loss, acc = sess.run(
                [cost, accuracy],
                feed_dict={
                    x: X_test,
                    y: one_hot(y_test)
                }
            )
            test_losses.append(loss)
            test_accuracies.append(acc)
            #print("PERFORMANCE ON TEST SET: " +               "Batch Loss = {}".format(loss) +               ", Accuracy = {}".format(acc))

        step += 1

    print("Optimization Finished!")

    # Accuracy for test data

    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
            x: X_test,
            y: one_hot(y_test)
        }
    )

    test_losses.append(final_loss)
    test_accuracies.append(accuracy)

    #print("FINAL RESULT: " +       "Batch Loss = {}".format(final_loss) +       ", Accuracy = {}".format(accuracy))


    # ## Training is good, but having visual insight is even better:
    #
    # Okay, let's plot this simply in the notebook for now.

    # In[17]:


    # (Inline plots: )



    predictions = one_hot_predictions.argmax(1)

    return predictions.astype(float)+1, y_test.astype(float) + 1

