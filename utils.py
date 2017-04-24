from bs4 import BeautifulSoup as bs
import random, re, json, itertools
from collections import defaultdict
from gensim.models import KeyedVectors
import numpy as np
from scipy.stats import entropy
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Normalizer, FunctionTransformer
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score
from functools import partial

parens = re.compile(r'\([\w]+\)')
number = re.compile(r'[0-9]+')
diactric = re.compile(r'[^a-z ]+')
spaces = re.compile(r'\s{2,}')

Identity = partial(FunctionTransformer, lambda x: x)

def purge(x):
    x = x.lower()
    x = parens.sub('', x)
    x = number.sub(' falsenumber ', x)
    return spaces.sub(' ', diactric.sub('', x).rstrip(' ').lstrip(' '))

def load_data(keywords=['technology', 'design', 'entertainment'], purify=purge):
    with open('ted_en_kw_cont.json', 'r') as f:
        data = json.load(f)
        
    N = len(data)
    x, y = [], []
    for data_dict in data:
        
        x.append(purify(data_dict['content']))
        y.append(['1' if y in data_dict['keywords'] else '0' for y in keywords])
        
    dataset = list(zip(x, y))
    random.shuffle(dataset)
    return dataset

def transform_labels_usable(y_set, keywords=['technology', 'design', 'entertainment']):
    
    nr_of_labels = 2**len(y_set[0])
    
    lst = itertools.product(['0', '1'], repeat=3)
    
    binary_to_nr = {}
    
    labels = []
    
    for i, rep in enumerate(lst):
        
        binary_to_nr[''.join(rep)] = i
        labels.append('_'.join([keywords[i] for i, x in enumerate(rep) if x == '1'])) 
        
    return list(map(lambda x: binary_to_nr[''.join(x)], y_set)), labels

def count_words(iterator):
    
    word_count_dict = defaultdict(int)
    
    for sequence in iterator:
        
        sequence = set(sequence.split())
        for word in sequence:
            word_count_dict[word] += 1
            
    return word_count_dict

def calculate_probs_occurence(words, texts, labels):
    
    N = len(texts)
    
    labels_dict = dict(zip(words, np.zeros((len(words), max(labels)+1))))
    
    for i in range(N):
        
        text = set(texts[i].split())
        
        for word in words:
            
            if word in text:
                
                labels_dict[word][labels[i]] += 1
                
    return labels_dict

def calculate_lab_prob(labels):
    
    lab, counts = np.unique(labels, return_counts=True)
    return counts/np.sum(counts)

def dense_transformer():
    return FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)

def test_score_one_vs_all(y_true, y_pred, score_func):
    
    labels = np.unique(y_true)
    scores = []
    
    for lab in labels:
        
        pred = (y_pred==lab).astype(int)
        true = (y_true==lab).astype(int)
        
        scores.append(score_func(y_pred=pred, y_true=true))
        
    return np.asarray(scores)

def get_score(clf, X, y, cv=10):
    
    N = len(X)
    s = N//cv
    
    #X, y = list(map(np.asarray, [X,y]))
    
    scores = defaultdict(list)
    
    for i in range(cv):
        
        x_train, x_test, y_train, y_test = train_test_split(X, y)
        
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        probs = clf.predict_proba(x_test)
        
        scores['log loss'].append( log_loss( y_test, probs ) )
        scores['accuracy'].append( accuracy_score( y_test, pred ) )
        scores['precision'].append(test_score_one_vs_all(y_test, pred, precision_score).mean())
        scores['recall'].append(test_score_one_vs_all(y_test, pred, recall_score).mean())
        
    return {k:np.mean(v) for k, v in scores.items()}
    
def train_mlc(vectorizer, model,  data, vocabulary, cv=10, sample_weight=None, **kwargs):
    
    pip = make_pipeline(vectorizer(vocabulary=vocabulary), 
                        dense_transformer(),
                        model(**kwargs)
                       )
    
    scores = defaultdict(list)
    X, y = data
    C = np.asarray(y).shape[1]
    for i in range(cv):
        
        x_train, x_test, y_train, y_test = train_test_split(X, y)
        
        logs = []
        accs = []
        precs = []
        recs = []
        
        predictions = []
        
        for c in range(C):
            
            pip.fit(x_train, y_train[:, c])
            pred = pip.predict(x_test)
            probs = pip.predict_proba(x_test)
            
            predictions.append(pred)
            
            true = y_test[:, c]
            
            logs.append( log_loss( true, probs ) )
            accs.append( accuracy_score( true, pred ))
            precs.append( precision_score( true, pred ))
            recs.append( recall_score( true, pred ))
            
        
        scores['accuracy'].append( np.mean( np.prod( y_test == np.asarray(predictions).T , axis=1)).mean() )
        scores['log loss'].append( np.mean( logs ))
        scores['partial accuracy'].append( np.mean( accs ))
        scores['precision'].append( np.mean( precs ))
        scores['recall'].append( np.mean( recs ))
        
    return {k:np.mean(v) for k, v in scores.items()}
    
    
def train_test_pipeline(vectorizer, 
                        model, 
                        data, 
                        vocabulary, 
                        sample_weight=None,
                        **kwargs):
    
    pip = make_pipeline(vectorizer(vocabulary=vocabulary), 
                        dense_transformer(),
                        model(**kwargs)
                       )
    
    X, y = data
    return get_score(pip, X, y), pip


def feature_selection_pipeline(vectorizer, 
                               model, 
                               data, 
                               selection_func, 
                               k, 
                               vocabulary=None, 
                               preprocessor=Identity, 
                               sample_weight=None,
                               **kwargs):
    
    
    pip = make_pipeline(vectorizer(vocabulary=vocabulary),
                        dense_transformer(),
                        SelectKBest(selection_func, k=k),
                        preprocessor()
                       )
    
    X, y = data
    X = pip.fit_transform(X, y)
    return get_score(model(**kwargs), np.asarray(X), np.asarray(y)), pip

class DocVectorizer(TransformerMixin):
    
    def __init__(self, keys, word_counts=None, r=1e-3, mean=True, normalize=True):
        
        self.mean = mean
        self.r = r
        
        if isinstance(keys, str):
            
            self.keys = KeyedVectors.load_word2vec_format(keys)
            
        elif isinstance(keys, KeyedVectors):
            
            self.keys = keys
            
        else:
            
            raise TypeError('give me KeyedVectors object or name of file')
            
        if word_counts is not None:
            words_sum = sum(word_counts.values())

            self.word_counts = {k: v for k, v in word_counts.items()}
            if normalize:
                self.word_counts = {k: self.word_counts[k]/words_sum for k in word_counts.keys()}
        
        else:
            self.word_counts = dict(zip(keys.index2word, [1e-9]*len(keys.index2word)))
        
    def _translate(self, X, y=None):
        
        all_embeds = []
        proper_inds = []
        
        for i, doc in enumerate(X):
            
            vecs = []
            
            if isinstance(doc, str):
                doc = doc.split()
                
            for word in doc:
                
                if word in self.word_counts and word in self.keys:
                    
                    if self.word_counts[word] > self.r and np.random.uniform() < 1 - self.r/self.word_counts[word]:
                        continue
                        
                    vecs.append( self.keys[word] )
            
            if not len(vecs):
                continue
                
            if self.mean:
                vecs = np.mean(np.asarray(vecs), axis=0)
                
            proper_inds.append(i)
            
            all_embeds.append(vecs)
            
            
        return (np.asarray(all_embeds), [y[i] for i in proper_inds]) if y is not None else all_embeds
    
    def fit(self, X, y=None):
        
        return self
    
    def fit_transform(self, X, y=None):
        
        return self._translate(X, y)
    
    def transform(self, X, y=None):
        
        return self._translate(X, y)