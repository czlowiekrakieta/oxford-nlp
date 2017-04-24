import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from sklearn.base import ClassifierMixin
from tensorflow.contrib.rnn import LSTMCell, GRUCell, LSTMStateTuple


def init_normal_var(shape):
    return tf.Variable(tf.truncated_normal(shape=shape))

def init_xavier(shape):
    bound = np.sqrt(6./sum(shape))
    return tf.Variable(tf.random_uniform(shape=shape, minval=-bound, maxval=bound))


class Model(ClassifierMixin):
    
    def __init__(self, what, iters=1000, batch=64, lr=1e-1, class_balance=True, alpha=0, 
                 reset_with_new_fit = True, x_test=None, y_test=None, **kwargs):
        
        params = dict(what=what, iters=iters, batch=batch, lr=lr, class_balance=class_balance, alpha=alpha, 
                 reset_with_new_fit = reset_with_new_fit, x_test=x_test, y_test=y_test)
        params.update(kwargs)

        
        self.params = params
        self.models = {'mlp':build_mlp,
                      'rnn':build_rnn}
        
        self.what = what
        
    
    def _real_init(self, what, iters=1000, batch=64, lr=1e-1, class_balance=True, alpha=0, 
                 reset_with_new_fit = True, x_test=None, y_test=None, **kwargs):
        
        tf.reset_default_graph()
        
        print(kwargs)
        
        
        params = dict(what=what, iters=iters, batch=batch, lr=lr, class_balance=class_balance, alpha=alpha, 
                 reset_with_new_fit = reset_with_new_fit, x_test=x_test, y_test=y_test)
        self.params.update(kwargs)
        
        
        self.placeholders, self.logits, self.stuff, self.classes = self.models[what](**kwargs)
        
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
        
        self.reset = reset_with_new_fit
        
    def step_fit(self, X, y, weights=None):
        
        labels, lab_count = np.unique(y, return_counts=True)
        N = len(X)
        C = len(labels)
        if weights is None:
            weights = np.asarray([N/float(C*lab_count[t]) for t in y])
        feed_dict = {}
        
        if 'lengths' in self.placeholders:
            
            E = len(X[0][0])
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
        C = max(labels)
        S = X[0].shape[0] if self.what == 'mlp' else len(X[0][0])
        
        self.params.update({'what':self.what, 'input_size':S, 'output_size':C+1})
        self._real_init(**self.params)
        
        weights = np.asarray([N/float(C*lab_count[t]) for t in y]) if self.balance else np.ones(N)
        if self.y_test is not None:
            self.test_weights = np.asarray([N/float(C*lab_count[t]) for t in self.y_test])
        while self.iters:
            
            B = random.sample(range(len(X)), self.batch)
            X, y = np.asarray(X), np.asarray(y)
            
            self.step_fit(X[B], y[B], weights[B])
            
            
            self.iters -= 1
            
        return self

    def predict(self, X):
        
        feed_dict={self.placeholders['inputs']:X, self.placeholders['training']:False}
        return self.sess.run(self.predictions, feed_dict=feed_dict)
    
    def predict_proba(self, X):
        
        feed_dict={self.placeholders['inputs']:X, self.placeholders['training']:False}
        
        return self.sess.run(self.probs, feed_dict=feed_dict)
        
    
    def score(self, X, y):
        
        feed_dict={self.placeholders['inputs']:X, self.placeholders['training']:False, self.placeholders['outputs']:y}
        preds = self.sess.run(self.predictions, feed_dict=feed_dict)
        return (preds == y).mean()
    
    def get_params(self, deep=True):
        
        return self.params
            
    def set_params(self, **kwargs):
        
        
        if hasattr(self, 'params'):
            self.params.update(kwargs)
        
        else:
            self.params = kwargs
        
        return self
    
def build_mlp(input_size, output_size, architecture, activation='relu', dropout=.6):
    
    inputs = tf.placeholder(tf.float32, [None, input_size])
    outputs = tf.placeholder(tf.int32, [None])
    
    training = tf.placeholder(tf.bool, [])
    
    if isinstance(architecture, int):
        architecture = [architecture]
    
    architecture = [input_size] + architecture + [output_size]
    
    activs = {
        'relu': tf.nn.relu,
        'sigmoid': tf.nn.sigmoid,
    }
    
    N = len(architecture)
        
    weights = [(
        init_xavier(architecture[i-1:i+1]),
        init_xavier([architecture[i]]))
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

def build_rnn(input_size, output_size, hidden, cell='lstm', average=False, bidirectional=True, time_major=False):
    
    inputs = tf.placeholder(tf.float32, [None, None, input_size])
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
        
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            sequence_length=lengths,
            inputs=inputs,
            dtype=tf.float32
        )
        
        if isinstance(states[0], LSTMStateTuple):
            state = LSTMStateTuple(c=tf.concat((states[0].c, states[1].c), axis=1), 
                                   h=tf.concat((states[0].h, states[1].h), axis=1))
        else:
            state = tf.concat((states[0], states[1]), axis=1)
            
        outputs = tf.concat((outputs[0], outputs[1]), axis=2)
        
    else:
        
        cell_fw = cells[cell](hidden)
        
        outputs, state = tf.nn.dynamic_rnn(
            cell=cell_fw,
            sequence_length=lengths,
            inputs=inputs,
            dtype=tf.float32
        )
        
    if isinstance(state, LSTMStateTuple):
        
        state = tf.concat((state.c, state.h), axis=1)

    if average:
        output = tf.reduce_sum(outputs, axis=1)/tf.expand_dims(tf.cast(lengths, tf.float32), axis=1)
        
        print(output)
        
        weights = init_xavier([output.get_shape()[1].value, output_size])
        bias = init_xavier([output_size])
        
        logits = tf.matmul(output, weights)+bias
        return {'inputs':inputs, 'outputs':targets, 'lengths':lengths, 'training':training}, logits, {'weights':[weights, bias], 'outputs':outputs}, output_size
        
    else:
        weights = init_normal_var([state.get_shape()[1].value, output_size])
        bias = init_normal_var([output_size])
        logits = tf.matmul(state, weights)+bias

        return {'inputs':inputs, 'outputs':targets, 'lengths':lengths, 'training':training}, logits, {'weights':[weights, bias], 'outputs':outputs}, output_size

    