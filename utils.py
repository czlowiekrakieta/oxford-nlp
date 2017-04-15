from bs4 import BeautifulSoup as bs
import random, re, json, itertools
from collections import defaultdict
import numpy as np
from scipy.stats import entropy
from sklearn.pipeline import make_pipeline
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Normalizer, FunctionTransformer
from sklearn.metrics import precision_score, recall_score
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

def load_and_train_test_split(keywords=['technology', 'design', 'entertainment'], train_size=.7, purify=purge):
    with open('ted_en_kw_cont.json', 'r') as f:
        data = json.load(f)
        
    N = len(data)
    x, y = [], []
    for data_dict in data:
        
        x.append(purify(data_dict['content']))
        y.append(['1' if y in data_dict['keywords'] else '0' for y in keywords])
        
    dataset = list(zip(x, y))
    random.shuffle(dataset)
    return dataset[:int(N*train_size)], dataset[int(N*train_size):]

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


def vectorize_docs(documents, word2vec, mean=True):
    
    vectors = []
    indices = []
    for i, doc in enumerate(documents):
        
        bag = [word2vec[x] for x in doc.split() if x in word2vec]
        if len(bag) == 0:
            continue
        indices.append(i)
        vectors.append(np.mean(bag, axis=0) if mean else np.asarray(bag))
        
    return np.vstack(vectors) if mean else vectors, indices

def dense_transformer():
    return FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)

def test_score_one_vs_all(y_pred, y_true, score_func):
    
    labels = np.unique(y_true)
    scores = []
    
    for lab in labels:
        
        pred = (y_pred==lab).astype(int)
        true = (y_true==lab).astype(int)
        
        scores.append(score_func(y_pred=pred, y_true=true))
        
    return np.asarray(scores)

def get_score(pip, x_train, y_train, x_test, y_test):
    preds = pip.predict(x_test)
    return {'train':pip.score(x_train, y_train), 'test':pip.score(x_test, y_test), 
            'precision':test_score_one_vs_all(preds, y_test, precision_score).mean(), 
            'recall':test_score_one_vs_all(preds, y_test, recall_score).mean()}, pip
    

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
    
    x_train, y_train, x_test, y_test = data
    
    pip.fit(x_train, y_train)
    return get_score(pip, x_train, y_train, x_test, y_test)


def feature_selection_pipeline(vectorizer, 
                               model, 
                               data, 
                               selection_func, 
                               k, 
                               vocabulary=None, 
                               preprocessor=Identity, 
                               sample_weight=None,
                               **kwargs):
    
    x_train, y_train, x_test, y_test = data
    
    pip = make_pipeline(vectorizer(vocabulary=vocabulary),
                        dense_transformer(),
                        SelectKBest(selection_func, k=k),
                        preprocessor(),
                        model(**kwargs)
                       )
    
    pip.fit(x_train, y_train, **{pip.steps[-1][0] + '__sample_weight':sample_weight})
    return get_score(pip, x_train, y_train, x_test, y_test)

def uninformed_train_pipeline(model,
                          data,
                          preprocessor,
                          sample_weight=None,
                          **kwargs
                             ):

    x_train, y_train, x_test, y_test = data
    
    pip = make_pipeline(preprocessor(),
                        model(**kwargs)
                       )
    
    pip.fit(x_train, y_train)
    return get_score(pip, x_train, y_train, x_test, y_test)


def prepare_data_to_rnn(data, word2vec):
    
    x_train, indices = vectorize_docs(data[0], word2vec, mean=False)
    y_train = np.asarray(data[1])[indices]
    
    x_test, indices = vectorize_docs(data[2], word2vec, mean=False)
    y_test = np.asarray(data[3])[indices]
    
    return x_train, y_train, x_test, y_test

def train_with_loss_history(vectorizer,
                            data,
                            tf_model,
                            vocabulary,
                            **kwargs):
    
    x_train, y_train, x_test, y_test = data
    pip = make_pipeline(vectorizer(vocabulary=vocabulary), 
                        dense_transformer()
                       )
    
    pip.fit(x_train)
    x_test = pip.transform(x_test)
    mod = tf_model(x_test=x_test, y_test=y_test, **kwargs)
    mod.fit(pip.transform(x_train), y_train)
    return get_score(mod, pip.transform(x_train), y_train, x_test, y_test)
    