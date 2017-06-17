import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell, rnn

from SENTIMENT import helpers

INPUT_SIZE = 1
STEP_SIZE = 10
HIDDEN_SIZE = 10
NUMCLASSES = 1

EPOCH = 50
BATCHSIZE = 5

ROWS_TOTAL = 830
FEATURES = 10  # 30 (x,y) pairs
TRAIN_SIZE = 660
TEST_SIZE = 170

learning_rate = 0.01

def shuffle(data, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_data = dataset[permutation,:]
    shuffled_labels = labels[permutation]
    return shuffled_data, shuffled_labels


def input_fnc_train(data, labels):
    train_data = dataset[0:TRAIN_SIZE,:].astype(np.int32)
    train_labels = labels[0:TRAIN_SIZE].astype(np.int32)  
    return train_data, train_labels


def input_fnc_test(data, labels):
    start = TRAIN_SIZE 
    end = start + TEST_SIZE
    test_data = dataset[start:end,:].astype(np.int32)
    test_labels = labels[start:end].astype(np.int32)
    return test_data, test_labels


def model(x):
    layer = {'weights':tf.Variable(tf.random_normal
                                    ([HIDDEN_SIZE,NUMCLASSES])),
            'biases':tf.Variable(tf.random_normal([NUMCLASSES]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1,INPUT_SIZE])
    x = tf.split(0,STEP_SIZE,x)
        
    cell = rnn_cell.BasicLSTMCell(HIDDEN_SIZE,state_is_tuple=True)
    outputs, states = rnn.rnn(cell, x, dtype=tf.float32)
    
    output = tf.matmul(outputs[-1],layer['weights'])+layer['biases']
    
    return output





dataset, labels = helpers.load_data()
dataset = dataset.as_matrix()
labels = labels.values
shuffled_data, shuffled_labels = shuffle(dataset, labels)

train_t, tf_train_labels = input_fnc_train(shuffled_data, shuffled_labels)
print(train_t)
tf_train = tf.constant(train_t)

test_t, tf_test_labels = input_fnc_test(shuffled_data, shuffled_labels)
print(test_t)
tf_test = tf.constant(test_t)
with tf.Graph().as_default():
    x = tf.placeholder('float', [None, STEP_SIZE, INPUT_SIZE])
    y = tf.placeholder('float', [None, NUMCLASSES])

    prediction = model(x)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction,y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    #saver = tf.train.Saver()
   

    with tf.Session() as sess:
        sess.run( tf.initialize_all_variables())
        
        #saved = saver.save(sess, SAVEPATH)
        
        print('Initialized')
        for epoch in range(EPOCH):
            epoch_loss = 0
            i = 0
            while i < TRAIN_SIZE:
                start = i;
                end = i+BATCHSIZE
            
                batch_x = train_t[start:end, :]
                batch_y = tf_train_labels[start:end, :]
                batch_x = batch_x.reshape((BATCHSIZE, STEP_SIZE, INPUT_SIZE))
                _, l = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += l
                i+=BATCHSIZE
            if(epoch % 3 == 0):
                print('Epoch', epoch, 'completed out of', EPOCH, 'loss: ', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
    
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:test_t.reshape((-1,STEP_SIZE,INPUT_SIZE)), y:tf_test_labels}))
