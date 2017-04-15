import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from sklearn.base import ClassifierMixin
from tensorflow.contrib.rnn import LSTMCell, GRUCell, LSTMStateTuple, static_rnn, static_bidirectional_rnn


def init_normal_var(shape):
    return tf.Variable(tf.truncated_normal(shape=shape))

class Model(ClassifierMixin):
    
    def __init__(self, func, iters=1000, batch=64, lr=1e-1, class_balance=True, alpha=0, x_test=None, y_test=None, **kwargs):
        
        tf.reset_default_graph()
        
        self.placeholders, self.logits, self.stuff, self.classes = func(**kwargs)
        
        self.targets_ph = self.placeholders['outputs']
        
        self.predictions = tf.cast(tf.argmax(self.logits, axis=1), dtype=tf.int32)
        
        self.probs = tf.nn.softmax(self.logits)
        
        
        self.weights = tf.placeholder('float', [None])
        
        self.loss = tf.reduce_mean(
            tf.multiply( 
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets_ph),
                self.weights
            )
        )
        
        if alpha > 0:
            for var in tf.trainable_variables():
                self.loss += alpha*tf.nn.l2_loss(var)
        if alpha < 0:
            raise ValueError('alpha parameter should larger than zero')
                                     
        self.opt = tf.train.AdamOptimizer(lr)
        
        self.grads_and_vars = self.opt.compute_gradients(self.loss)
        
        self.applier = self.opt.apply_gradients(self.grads_and_vars)
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        
        self.iters = iters
        self.batch = batch
        self.balance = class_balance
        
        self.x_test = x_test
        self.y_test = y_test
        
        self.train_loss = []
        self.test_loss = []
        
    def step_fit(self, X, y, weights=None):
        
        labels, lab_count = np.unique(y, return_counts=True)
        N = len(X)
        C = len(labels)
        if weights is None:
            weights = np.asarray([N/float(C*lab_count[t]) for t in y])
        feed_dict = {}
        
        if 'lengths' in self.placeholders:
            
            E = X[0].shape[1]
            L = list(map(len, X))
            M = max(L)
            temp = np.zeros((N, M, E))
            
            for i in range(N):
                
                temp[i] = np.concatenate([X[i], np.zeros((M-L[i], E))], axis=0)
                
                
            X = np.asarray(temp)
            feed_dict = {self.placeholders['lengths']:L}
                
        feed_dict.update({self.placeholders['inputs']:X, self.placeholders['outputs']:y, self.weights:weights, 
                         self.placeholders['training']:True})
        
        self.sess.run(self.applier, feed_dict=feed_dict)
        if self.x_test is not None:
            feed_dict[self.placeholders['training']] = False
            loss = self.sess.run(self.loss, feed_dict=feed_dict)
            feed_dict.update({self.placeholders['inputs']:self.x_test, self.placeholders['outputs']:self.y_test,
                              self.weights:self.test_weights})
            test_loss = self.sess.run(self.loss, feed_dict)
            self.train_loss.append(loss)
            self.test_loss.append(test_loss)
                
    def fit(self, X, y):
        import random
        N = len(X)
        labels, lab_count = np.unique(y, return_counts=True)
        C = len(labels)
        
        weights = np.asarray([N/float(C*lab_count[t]) for t in y]) if self.balance else np.ones(N)
        if self.y_test is not None:
            self.test_weights = np.asarray([N/float(C*lab_count[t]) for t in self.y_test])
        while self.iters:
            
            B = random.sample(range(len(X)), self.batch)
            X, y = np.asarray(X), np.asarray(y)
            
            self.step_fit(X[B], y[B], weights[B])
            
            
            self.iters -= 1

    def predict(self, X):
        
        feed_dict={self.placeholders['inputs']:X, self.placeholders['training']:False}
        return self.sess.run(self.predictions, feed_dict=feed_dict)
    
    def score(self, X, y):
        
        feed_dict={self.placeholders['inputs']:X, self.placeholders['training']:False}
        preds = self.sess.run(self.predictions, feed_dict=feed_dict)
        return (preds == y).mean()
        
def build_mlp(input_size, output_size, architecture, activation='relu', dropout=.6):
    
    inputs = tf.placeholder(tf.float32, [None, input_size])
    outputs = tf.placeholder(tf.int32, [None])
    
    training = tf.placeholder(tf.bool, [])
    
    architecture = [input_size] + architecture + [output_size]
    
    activs = {
        'relu': tf.nn.relu,
        'sigmoid': tf.nn.sigmoid,
    }
    
    N = len(architecture)
        
    weights = [(
        init_normal_var(architecture[i-1:i+1]),
        init_normal_var([architecture[i]]))
        for i in range(1, N)
    ]

    progression_of_layers = [inputs]
        
    for i in range(N-2):

        progression_of_layers.append( 
            tf.matmul(progression_of_layers[-1], weights[i][0]) + weights[i][1]
        )
        progression_of_layers.append(
            activs[activation](progression_of_layers[-1])
        )
        progression_of_layers.append(
            tf.contrib.layers.dropout(progression_of_layers[-1], keep_prob=dropout, is_training=training)
        )
        
        
    logits = tf.matmul(progression_of_layers[-1], weights[N-2][0]) + weights[N-2][1]
    
    return {'inputs':inputs, 'outputs':outputs, 'training':training}, logits, {'weights': weights, 'layers': progression_of_layers}, output_size

def build_rnn(input_size, output_size, hidden, cell='lstm', bidirectional=True, time_major=False, max_length=6395):
    
    inputs = tf.placeholder(tf.float32, [None, max_length, input_size])
    targets = tf.placeholder(tf.int32, [None])
    training = tf.placeholder(tf.bool, [])
    
    cells = {
        'lstm': LSTMCell,
        'gru': GRUCell,
    }
    
    lengths = tf.placeholder(tf.int32, [None])
    outputs, state = [], []
    
    if bidirectional:
        
        cell_fw = cells[cell](hidden)
        cell_bw = cells[cell](hidden)
        
        outputs, states = static_bidirectional_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            sequence_length=lengths,
            inputs=tf.unstack(inputs, axis=1),
            dtype=tf.float32
        )
        
        if isinstance(states[0], LSTMStateTuple):
            state = LSTMStateTuple(c=tf.concat((states[0].c, states[1].c), axis=1), 
                                   h=tf.concat((states[0].h, states[1].h), axis=1))
        else:
            state = tf.concat((states[0], states[1]), axis=1)
        
    else:
        
        cell_fw = cells[cell](hidden)
        
        outputs, state = static_rnn(
            cell=cell_fw,
            sequence_length=lengths,
            inputs=tf.unstack(inputs, axis=1),
            dtype=tf.float32
        )
        
    if isinstance(state, LSTMStateTuple):
        
        state = tf.concat((state.c, state.h), axis=1)

    weights = init_normal_var([2*hidden if cell=='lstm' else hidden, output_size])
    bias = init_normal_var([output_size])
    logits = tf.matmul(state, weights)+bias
    
    return {'inputs':inputs, 'outputs':targets, 'lengths':lengths, 'training':training}, logits, {'weights':[weights, bias], 'outputs':outputs}, output_size