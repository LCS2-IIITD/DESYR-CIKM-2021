'''
Colab Link: https://colab.research.google.com/drive/1KrRUvI-dhNbRBKUmWeJcww0Qy078-vDu?usp=sharing
'''

import pandas as pd
import numpy as np
import os
import re
import pickle
import preprocessor as p
import nltk
from nltk.probability import FreqDist
from collections import Counter
from itertools import chain
from scipy.stats import mode 
from tqdm.notebook import tqdm

import spacy
nlp = spacy.load('en_core_web_sm')

from imblearn.under_sampling import RandomUnderSampler

from focal_loss import SparseCategoricalFocalLoss

import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import(
    Model, initializers, regularizers, constraints,
    optimizers, layers
)
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Dense, Layer, Input, Lambda, Dot, Bidirectional, LSTM,
    TimeDistributed, Embedding, Reshape, Flatten, concatenate,
    Attention, GlobalAveragePooling1D, RepeatVector, Concatenate,
    ZeroPadding1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.framework import ops
from keras import backend as K

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')

class Dataset:
  def __init__(self, dataset, text_column):
    '''
    dataset: input dataframe
    text_column: name of column with the text 
    '''
    self.dataset = dataset
    self.text_column = text_column
  
  def decontracted(self, phrase):
    # SPECIFIC
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # GENERAL
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    
    return phrase
  
  def rem_special(self, phrase):
    alphanum_regex = r'[^a-zA-z0-9\s]'
    space_regex = r' +'

    phrase = re.sub(alphanum_regex, ' ', phrase)
    phrase = re.sub(space_regex, ' ', phrase)

    return phrase
  
  def clean_tweet(self, text):
    text = self.decontracted(text)
    text = p.clean(text)
    text = self.rem_special(text)
    text = text.lower()
    
    return text
  
  def get_dependency(self, text):
    dep_sentence = [token.dep_ for token in nlp(text)]
    return dep_sentence
  
  def get_pos(self, text):
    pos_sentence = ["{}.{}".format(token.text.lower(),token.pos_.lower()) for token in nlp(text)]
    return " ".join(pos_sentence)
  
  def preprocess(self, p_init=False, get_dep=True):
    self.dataset[self.text_column] = self.dataset.apply(lambda x: self.clean_tweet(x[self.text_column]), axis=1)
    self.dataset = self.dataset.drop_duplicates(subset=self.text_column, keep="last")
    if p_init:
      self.dataset[self.text_column] = self.dataset.apply(lambda x: self.get_pos(x[self.text_column]), axis=1)
    if get_dep:
      self.dataset['DEP'] = self.dataset.apply(lambda x: self.get_dependency(x[self.text_column]), axis=1)
    return self.dataset.sample(frac=1, random_state=0).reset_index(drop=True)

class WVPoincare:
  def __init__(self, filename):
    self.filename = filename
    self.model = self.loadModel()
  
  def loadModel(self):
    with open('gensim_poincare/' + self.filename, 'rb') as handle:
      model = pickle.load(handle, encoding='latin1')
    return model
  
  def getModel(self):
    return self.model
  
  def plotTSNE(self, size=None):
    labels = []
    tokens = []

    for token in tqdm(self.model.kv.vocab):
      tokens.append(self.model.kv[token])
      labels.append(token)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=0)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for val in tqdm(new_values):
      x.append(val[0])
      y.append(val[1])
    
    plt.figure(figsize=(20, 20), dpi=100) 
    for i in tqdm(range(len(x))):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

class TextDataLoader:
  def __init__(self, dataset, text_column, label_column, embedding_model, num_words, EMBEDDING_DIM=100, train_size=0.85):
    self.dataset = dataset
    self.text_column = text_column
    self.label_column = label_column
    self.embedding_model = embedding_model
    self.num_words = num_words
    self.EMBEDDING_DIM = EMBEDDING_DIM
    self.train_size = train_size
  
  def get_embeddingsindex(self, filename, return_index=False, embedding_type=None):
    embeddings_index = {}
    if embedding_type is None:
      f = open(os.path.join('gensim_poincare',filename), encoding = "utf-8")
    elif embedding_type == 'glove':
      f = open('glove_100.txt', encoding = "utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embeddings_index[word] = coefs
    f.close
    
    self.embeddings_index = embeddings_index
    if return_index:
      return self.embeddings_index
  
  def save_tokenizer(self, pickle_name, tokenizer):
    with open(pickle_name, 'wb') as handle:
      pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
  def load_tokenizer(self, pickle_name):
    with open(pickle_name, 'rb') as handle:
      tokenizer = pickle.load(handle)
    
    return tokenizer
  
  def get_tokenizer(self, return_tokenizer=False, stream='tw'):
    try:
      tokenizer = self.load_tokenizer("tokenizer_{}.pickle".format(stream))
      print('Found %s unique tokens.' % len(tokenizer.word_index))
    except:
      tokenizer = Tokenizer(num_words=self.num_words, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n', lower=True, oov_token='UNK')
      tokenizer.fit_on_texts(self.dataset[self.text_column][:int(self.train_size*len(self.dataset))])
      print('Found %s unique tokens.' % len(tokenizer.word_index))
      self.save_tokenizer("tokenizer_{}.pickle".format(stream), tokenizer=tokenizer)
    
    self.tokenizer = tokenizer

    if return_tokenizer:
      return self.tokenizer
  
  def get_embeddingmatrix(self, return_embeddingmatrix=False):
    word_index = self.tokenizer.word_index

    vocab_size = len(word_index) + 1
    embedding_matrix = np.random.uniform(-1,1,(vocab_size,self.EMBEDDING_DIM))

    for word, i in word_index.items():
        embedding_vector = self.embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    self.vocab_size, self.embedding_matrix = vocab_size, embedding_matrix
    
    if return_embeddingmatrix:
      return self.vocab_size, self.embedding_matrix
  
  def get_data(self):
    return self.dataset[self.text_column].values, self.dataset[self.label_column].values

class TextDataGenerator:
  def __init__(self, texts, labels, tokenizer, maxlen=50, train_size=0.85, validation_split=0.10, sampling_strategy=1.0):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.maxlen = maxlen
    self.train_size = train_size
    self.validation_split = validation_split
    self.sampling_strategy = sampling_strategy

    self.texts_tr, self.texts_ts, self.labels_tr, self.labels_ts = train_test_split(list(self.texts), list(self.labels), train_size=self.train_size, shuffle=False)
  
  def to_matrix(self, l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]
  
  def split(self, fold_id):
    self.undersample = RandomUnderSampler(sampling_strategy=self.sampling_strategy, random_state=fold_id)
    texts_tr_resampled, labels_tr_resampled = self.undersample.fit_resample(self.to_matrix(self.texts_tr,1), self.to_matrix(self.labels_tr,1))

    self.texts = list(chain.from_iterable(texts_tr_resampled)) + self.texts_ts
    self.labels = list(chain.from_iterable(labels_tr_resampled)) + self.labels_ts
    self.train_size = len(labels_tr_resampled) / len(self.labels)

    texts_tr, texts_ts, labels_tr, labels_ts = train_test_split(
        list(self.texts), list(self.labels), train_size=self.train_size, shuffle=False
    )

    texts_tr, texts_val, labels_tr, labels_val = train_test_split(
        list(texts_tr), list(labels_tr), train_size=None, test_size=self.validation_split, shuffle=True, stratify=labels_tr, random_state=fold_id
    )

    train_generator = {
        "X": texts_tr,
        "y": labels_tr
    }

    validation_generator = {
        "X": texts_val,
        "y": labels_val
    }

    test_generator = {
        "X": texts_ts,
        "y": labels_ts
    }

    return train_generator, validation_generator, test_generator
  
  def get_generator(self, fold_id):
    train_generator, validation_generator, test_generator = self.split(fold_id=fold_id)

    train_generator['X'] = self.tokenizer.texts_to_sequences(train_generator['X'])
    train_generator['X'] = pad_sequences(
        train_generator['X'], maxlen=self.maxlen, padding='post', truncating='post'
    )

    validation_generator['X'] = self.tokenizer.texts_to_sequences(validation_generator['X'])
    validation_generator['X'] = pad_sequences(
        validation_generator['X'], maxlen=self.maxlen, padding='post', truncating='post'
    )

    test_generator['X'] = self.tokenizer.texts_to_sequences(test_generator['X'])
    test_generator['X'] = pad_sequences(
        test_generator['X'], maxlen=self.maxlen, padding='post', truncating='post'
    )

    return train_generator, validation_generator, test_generator
  
  def text_encoder(self, texts):
    encoded_texts = self.tokenizer.texts_to_sequences(texts)
    encoded_texts = pad_sequences(
        encoded_texts, maxlen=self.maxlen, padding='post', truncating='post'
    )

    return encoded_texts

class Attention(Layer):
    
    def __init__(self, return_sequences=True, name=None):
        self.return_sequences = return_sequences
        super(Attention,self).__init__(name=name)
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(Attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return K.cast(output, K.floatx())
        
        return K.sum(output, axis=1)

@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class GradientReversal(Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)

class EvaluateTestSet(Callback):
  def __init__(self, save_location, test_input, y_test, restore=True):
    super(EvaluateTestSet, self).__init__()
    self.save_location = save_location
    self.test_input = test_input
    self.y_test = y_test
    self.restore = restore

    self.best_weights = None
  
  def on_train_begin(self, logs=None):
    self.best = 0
  
  def on_epoch_end(self, epoch, logs=None):
    metrics = self.model.predict(
        self.test_input
    )
    y_pred = [np.argmax(el) for el in metrics[1]]
    current = f1_score(self.y_test, y_pred, average='macro')
    if np.greater(current, self.best):
      self.best = current
      self.best_weights = self.model.get_weights()

      print("saving model to {}".format(self.save_location))
      self.model.save_weights(self.save_location)
    else:
      if self.restore:
        print("Restoring model weights from the end of the best epoch")
        self.model.set_weights(self.best_weights)

def f1_macro(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def most_common(inp_list):
  val, count = mode(inp_list, axis = 0)
  return val.ravel()

class MyModel:
  def __init__(self, definitions, aux_weight=1.0, main_weight=1.0, config=None, vocab_size=None, EMBEDDING_DIM=None, embedding_matrix=None, maxlen=None, lstm_hidden_units=256, batch_size=16):
    self.aux_weight = aux_weight
    self.main_weight = main_weight
    self.vocab_size = vocab_size
    self.EMBEDDING_DIM = EMBEDDING_DIM
    self.embedding_matrix = embedding_matrix
    self.maxlen = maxlen
    self.lstm_hidden_units = lstm_hidden_units
    self.batch_size = batch_size

    self.c_definitions = definitions['claim']
    self.len_c_definitions = len(self.c_definitions)

    self.nc_definitions = definitions['non-claim']
    self.len_nc_definitions = len(self.nc_definitions)

    self.slice_lt = Lambda(lambda x: x[:,:self.lstm_hidden_units*2])
    self.slice_rt = Lambda(lambda x: x[:,self.lstm_hidden_units*2:])
    
    if config:
      self.config = config
    else:
      self.config = {
          'vocab_size': self.vocab_size,
          'EMBEDDING_DIM': self.EMBEDDING_DIM,
          'embedding_matrix': self.embedding_matrix,
          'maxlen': self.maxlen
      }
  
  def definition_network(self, inp, emb_layer, query_embeddings, model_id, type_def):
    if type_def == 'claim':
      inp_c_definitions = {}
      for i in range(self.len_c_definitions):
        inp_c_definitions[i] = Input(shape=(self.config['maxlen'],), name='inp_c_definition_{}_{}'.format(i, model_id))
      inp_definitions = inp_c_definitions

    if type_def == 'non-claim':
      inp_nc_definitions = {}
      for i in range(self.len_nc_definitions):
        inp_nc_definitions[i] = Input(shape=(self.config['maxlen'],), name='inp_nc_definition_{}_{}'.format(i, model_id))
      inp_definitions = inp_nc_definitions
    
    lstm_encoder = Bidirectional(LSTM(64, dropout=0.25, return_sequences=True))
    dense_encoder = Dense(32, activation="relu", name='definition_dnn_{}.{}'.format(type_def, model_id))
    definition_map = None
    for i in range(len(inp_definitions)):
      value_embeddings = emb_layer(inp_definitions[i])
      query_seq_encoding = lstm_encoder(query_embeddings)
      value_seq_encoding = lstm_encoder(value_embeddings)
      query_value_attention_seq = tf.keras.layers.Attention(use_scale=True)(
          [query_seq_encoding, value_seq_encoding]
      )
      query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
          query_seq_encoding
      ) 
      query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
          query_value_attention_seq
      )
      map = concatenate([query_encoding, query_value_attention])
      map = dense_encoder(map)
      if definition_map is None:
        definition_map = map
      else:
        definition_map = concatenate([definition_map, map])

    return definition_map, inp_definitions
  
  def lstm_network(self, model_id, induce_definition=True):
    inp = Input(shape=(self.config['maxlen'],), name='inp_{}'.format(model_id))

    emb_layer = Embedding(
        self.config['vocab_size'],
        self.config['EMBEDDING_DIM'],
        weights=[self.config['embedding_matrix']],
        mask_zero=True,
        trainable=True
    )

    # STANDARD ENCODING
    emb = emb_layer(inp)
    preds = Bidirectional(LSTM(256, dropout=0.25, return_sequences=True), name='bidirectional_1.{}'.format(model_id))(emb)
    preds = Bidirectional(LSTM(256, dropout=0.25, return_sequences=True), name='bidirectional_2.{}'.format(model_id))(preds)

    # DEFINITION MAP
    if induce_definition:
      definition_map_c, inp_c_definitions = self.definition_network(inp, emb_layer, emb, model_id, type_def='claim')
      definition_map_nc, inp_nc_definitions = self.definition_network(inp, emb_layer, emb, model_id, type_def='non-claim')

      definition_map_c = RepeatVector(self.maxlen)(definition_map_c)
      definition_map_nc = RepeatVector(self.maxlen)(definition_map_nc)

      preds = Concatenate(axis=2)(
          [preds, definition_map_c, definition_map_nc]
      )
      preds = TimeDistributed(Dense(2*self.lstm_hidden_units, activation="relu"))(preds)

      inputs = [
                inp, inp_c_definitions.values(), inp_nc_definitions.values()
      ]
    else:
      inputs = [inp]

    preds = Attention(return_sequences=False, name='attention_1.{}'.format(model_id))(preds)
    preds = Dense(2, activation="softmax")(preds)

    model = Model(
        inputs = inputs,
        outputs = [preds]
    )
    model.compile(
        optimizer ='adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy', f1_macro]
    )

    model.summary()

    return model
  
  def proj(self, tensor):
    x = tensor[0]
    y = tensor[1]
    return y * Dot(axes=1)([x, y]) / Dot(axes=1)([y, y])
  
  def orthogonalproj_layer(self, tensor):
    pnet_tensor = self.slice_lt(tensor)
    cnet_tensor = self.slice_rt(tensor)

    proj_pc = self.proj([pnet_tensor, cnet_tensor])
    return self.proj([pnet_tensor, pnet_tensor - proj_pc])

  def featureproj_model(self, pnet, cnet, custom_loss=None, return_model=False):
    cnet_features = cnet['model'].get_layer(cnet['layer_name']).output
    pnet_features = pnet['model'].get_layer(pnet['layer_name']).output

    flip_layer = GradientReversal()
    gr_out = flip_layer(cnet_features)
    gr_out = Attention(return_sequences=False)(gr_out)
    aux_out = Dense(2, activation='softmax', name='aux_output')(gr_out)

    features = concatenate([pnet_features, cnet_features])
    lambda_out = TimeDistributed(Lambda(self.orthogonalproj_layer, name="orthogonalproj_layer"))(features)
    lambda_out = Attention(return_sequences=False)(features)
    main_out = Dense(2, activation='softmax', name='main_output')(lambda_out)

    inputs = [
              cnet['model'].input, 
              pnet['model'].input    
    ]   
    outputs = [
               aux_out,
               main_out
    ]

    model = Model(
        inputs = inputs,
        outputs = outputs
    )
    if custom_loss is None: 
      model.compile(
          optimizer='adam',
          loss={
              'aux_output': 'sparse_categorical_crossentropy',
              'main_output': 'sparse_categorical_crossentropy'
          },
          loss_weights={
              'aux_output': self.aux_weight,
              'main_output': self.main_weight
          },
          metrics = ['accuracy', f1_macro]
      )
    elif custom_loss == 'focal':
      model.compile(
          optimizer='adam',
          loss={
              'aux_output': SparseCategoricalFocalLoss(gamma=2),
              'main_output': SparseCategoricalFocalLoss(gamma=2)
          },
          loss_weights={
              'aux_output': self.aux_weight,
              'main_output': self.main_weight
          },
          metrics = ['accuracy', f1_macro]
      )

    model.summary()

    self.model = model
    if return_model:
      return self.model

  def train(self, generators, fold_id, epochs=25, batch_size=16, model_prefix='_claim_lstm_orthogonalproj_def_focal', return_model=True, cw=False, es=True, cp=True, ets=True, ets_restore=True):
    self.generators = generators
    self.len_train = len(self.generators['train']['y'])
    self.len_validation = len(self.generators['validation']['y'])
    self.len_test = len(self.generators['test']['y'])
    
    if cw:
      class_weights = class_weight.compute_class_weight('balanced',
                                                        np.unique(self.generators['train']['y']),
                                                        self.generators['train']['y'])
      class_weights = {
          0: class_weights[0],
          1: class_weights[1]
      }
      print(class_weights)
    else:
      class_weights = None
    
    earlyStopping = EarlyStopping(monitor='val_main_output_f1_macro', min_delta=0, patience=10, verbose=0, mode='max')

    checkpoint = ModelCheckpoint("{}_{}.h5".format(model_prefix,fold_id), monitor='val_main_output_f1_macro', verbose=1,
                                 save_best_only=True, mode='max', period=1, save_weights_only=True)
    
    # CUSTOM CALLBACK
    save_location = "{}_{}.h5".format(model_prefix,fold_id)
    test_input = {
        "inp_1": np.array(self.generators['test']['X']), "inp_2": np.array(self.generators['test']['X']),
        "inp_c_definition_0_1": np.array([self.c_definitions[0]]*self.len_test), "inp_c_definition_1_1": np.array([self.c_definitions[1]]*self.len_test),
        "inp_nc_definition_0_1": np.array([self.nc_definitions[0]]*self.len_test), "inp_nc_definition_1_1": np.array([self.nc_definitions[1]]*self.len_test)
    }
    y_test = self.generators['test']['y']
    evaluateTestSet = EvaluateTestSet(
        save_location = save_location,
        test_input = test_input,
        y_test = y_test,
        restore = ets_restore
    )
    
    callbacks = []
    if es:
      callbacks.append(earlyStopping)
    if cp:
      callbacks.append(checkpoint)
    if ets:
      callbacks.append(evaluateTestSet)
    
    history = self.model.fit(
        {
            "inp_1": np.array(self.generators['train']['X']), "inp_2": np.array(self.generators['train']['X']),
            "inp_c_definition_0_1": np.array([self.c_definitions[0]]*self.len_train), "inp_c_definition_1_1": np.array([self.c_definitions[1]]*self.len_train),
            "inp_c_definition_2_1": np.array([self.c_definitions[2]]*self.len_train), "inp_c_definition_3_1": np.array([self.c_definitions[3]]*self.len_train),
            "inp_c_definition_4_1": np.array([self.c_definitions[4]]*self.len_train), "inp_c_definition_5_1": np.array([self.c_definitions[5]]*self.len_train),
            "inp_c_definition_6_1": np.array([self.c_definitions[6]]*self.len_train), "inp_c_definition_7_1": np.array([self.c_definitions[7]]*self.len_train),
            "inp_c_definition_8_1": np.array([self.c_definitions[8]]*self.len_train), "inp_c_definition_9_1": np.array([self.c_definitions[9]]*self.len_train),
            "inp_nc_definition_0_1": np.array([self.nc_definitions[0]]*self.len_train), "inp_nc_definition_1_1": np.array([self.nc_definitions[1]]*self.len_train),
            "inp_nc_definition_2_1": np.array([self.nc_definitions[2]]*self.len_train), "inp_nc_definition_3_1": np.array([self.nc_definitions[3]]*self.len_train),
            "inp_nc_definition_4_1": np.array([self.nc_definitions[4]]*self.len_train), "inp_nc_definition_5_1": np.array([self.nc_definitions[5]]*self.len_train),
            "inp_nc_definition_6_1": np.array([self.nc_definitions[6]]*self.len_train), "inp_nc_definition_7_1": np.array([self.nc_definitions[7]]*self.len_train),
        },
        {
            "aux_output": np.array(self.generators['train']['y']), "main_output": np.array(self.generators['train']['y'])
        },
        batch_size = batch_size,
        epochs=epochs,
        validation_data = (
            {
                "inp_1": np.array(self.generators['validation']['X']), "inp_2": np.array(self.generators['validation']['X']),
                "inp_c_definition_0_1": np.array([self.c_definitions[0]]*self.len_validation), "inp_c_definition_1_1": np.array([self.c_definitions[1]]*self.len_validation),
                "inp_c_definition_2_1": np.array([self.c_definitions[2]]*self.len_validation), "inp_c_definition_3_1": np.array([self.c_definitions[3]]*self.len_validation),
                "inp_c_definition_4_1": np.array([self.c_definitions[4]]*self.len_validation), "inp_c_definition_5_1": np.array([self.c_definitions[5]]*self.len_validation),
                "inp_c_definition_6_1": np.array([self.c_definitions[6]]*self.len_validation), "inp_c_definition_7_1": np.array([self.c_definitions[7]]*self.len_validation),
                "inp_c_definition_8_1": np.array([self.c_definitions[8]]*self.len_validation), "inp_c_definition_9_1": np.array([self.c_definitions[9]]*self.len_validation),
                "inp_nc_definition_0_1": np.array([self.nc_definitions[0]]*self.len_validation), "inp_nc_definition_1_1": np.array([self.nc_definitions[1]]*self.len_validation),
                "inp_nc_definition_2_1": np.array([self.nc_definitions[2]]*self.len_validation), "inp_nc_definition_3_1": np.array([self.nc_definitions[3]]*self.len_validation),
                "inp_nc_definition_4_1": np.array([self.nc_definitions[4]]*self.len_validation), "inp_nc_definition_5_1": np.array([self.nc_definitions[5]]*self.len_validation),
                "inp_nc_definition_6_1": np.array([self.nc_definitions[6]]*self.len_validation), "inp_nc_definition_7_1": np.array([self.nc_definitions[7]]*self.len_validation),
            },
            {
                "aux_output": np.array(self.generators['validation']['y']), "main_output": np.array(self.generators['validation']['y'])
            }
        ),
        callbacks = callbacks,
        class_weight = class_weights
    )

    self.model.load_weights("{}_{}.h5".format(model_prefix,fold_id))

    if return_model:
      return self.model
  
  def evaluate(self, batch_size=16):
    metrics = self.model.predict(
        {
            "inp_1": np.array(self.generators['test']['X']), "inp_2": np.array(self.generators['test']['X']),
            "inp_c_definition_0_1": np.array([self.c_definitions[0]]*self.len_test), "inp_c_definition_1_1": np.array([self.c_definitions[1]]*self.len_test),
            "inp_c_definition_2_1": np.array([self.c_definitions[2]]*self.len_test), "inp_c_definition_3_1": np.array([self.c_definitions[3]]*self.len_test),
            "inp_c_definition_4_1": np.array([self.c_definitions[4]]*self.len_test), "inp_c_definition_5_1": np.array([self.c_definitions[5]]*self.len_test),
            "inp_c_definition_6_1": np.array([self.c_definitions[6]]*self.len_test), "inp_c_definition_7_1": np.array([self.c_definitions[7]]*self.len_test),
            "inp_c_definition_8_1": np.array([self.c_definitions[8]]*self.len_test), "inp_c_definition_9_1": np.array([self.c_definitions[9]]*self.len_test),
            "inp_nc_definition_0_1": np.array([self.nc_definitions[0]]*self.len_test), "inp_nc_definition_1_1": np.array([self.nc_definitions[1]]*self.len_test),
            "inp_nc_definition_2_1": np.array([self.nc_definitions[2]]*self.len_test), "inp_nc_definition_3_1": np.array([self.nc_definitions[3]]*self.len_test),
            "inp_nc_definition_4_1": np.array([self.nc_definitions[4]]*self.len_test), "inp_nc_definition_5_1": np.array([self.nc_definitions[5]]*self.len_test),
            "inp_nc_definition_6_1": np.array([self.nc_definitions[6]]*self.len_test), "inp_nc_definition_7_1": np.array([self.nc_definitions[7]]*self.len_test),
        }
    )
    
    y_pred_bool = [np.argmax(el) for el in metrics[1]]
    print(classification_report(self.generators['test']['y'], y_pred_bool, target_names=['non-claim', 'claim']))
  
  def plot_roc(self, recall, precision, ix, y_test):
    plt.figure()

    y_test = np.array(y_test)
    no_skill = len(y_test[y_test==1]) / len(y_test)
    
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label='neural model')
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
  
  def calibrate_probabilities(self, predicted_proba, y_test):
    precision, recall, thresholds = precision_recall_curve(y_test, predicted_proba)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)

    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    self.plot_roc(recall, precision, ix, y_test)

    return thresholds[ix]
 
  def voting_classifier(self, generators=None, calibrate=None, error_analysis=False, print_preds=False, model_prefix='_claim_lstm_orthogonalproj_def_focal', num_folds=5):
    if generators is not None:
      self.generators = generators
      self.len_test = len(self.generators['test']['y'])

    test_inp = {
        "inp_1": np.array(self.generators['test']['X']), "inp_2": np.array(self.generators['test']['X']),
        "inp_c_definition_0_1": np.array([self.c_definitions[0]]*self.len_test), "inp_c_definition_1_1": np.array([self.c_definitions[1]]*self.len_test),
        "inp_c_definition_2_1": np.array([self.c_definitions[2]]*self.len_test), "inp_c_definition_3_1": np.array([self.c_definitions[3]]*self.len_test),
        "inp_c_definition_4_1": np.array([self.c_definitions[4]]*self.len_test), "inp_c_definition_5_1": np.array([self.c_definitions[5]]*self.len_test),
        "inp_c_definition_6_1": np.array([self.c_definitions[6]]*self.len_test), "inp_c_definition_7_1": np.array([self.c_definitions[7]]*self.len_test),
        "inp_c_definition_8_1": np.array([self.c_definitions[8]]*self.len_test), "inp_c_definition_9_1": np.array([self.c_definitions[9]]*self.len_test),
        "inp_nc_definition_0_1": np.array([self.nc_definitions[0]]*self.len_test), "inp_nc_definition_1_1": np.array([self.nc_definitions[1]]*self.len_test),
        "inp_nc_definition_2_1": np.array([self.nc_definitions[2]]*self.len_test), "inp_nc_definition_3_1": np.array([self.nc_definitions[3]]*self.len_test),
        "inp_nc_definition_4_1": np.array([self.nc_definitions[4]]*self.len_test), "inp_nc_definition_5_1": np.array([self.nc_definitions[5]]*self.len_test),
        "inp_nc_definition_6_1": np.array([self.nc_definitions[6]]*self.len_test), "inp_nc_definition_7_1": np.array([self.nc_definitions[7]]*self.len_test),
    }
    y_test = self.generators['test']['y']

    model_preds = []
    model_probs = []

    for i in range(num_folds):
      self.model.load_weights('{}_{}.h5'.format(model_prefix,i))
      metrics = self.model.predict(
          test_inp
      )
      predicted_proba = [el[1] for el in metrics[1]]
      model_probs.append(predicted_proba)

      if calibrate is not None:
        if calibrate['auto']:
          optimum_threshold = self.calibrate_probabilities(predicted_proba, y_test)
        else:
          optimum_threshold = calibrate['manual']
        model_preds.append([1 if proba > optimum_threshold else 0 for proba in predicted_proba])
      else:
        model_preds.append([np.argmax(el) for el in metrics[1]])

    final_pred = most_common(model_preds)
    print(len(final_pred))
    print(classification_report(y_test, final_pred, target_names=['non-claim', 'claim']))

    if error_analysis:
      misclassified_probs_nc = []
      misclassified_probs_c = []

      final_prob = np.mean(model_probs, axis=0)
      for prob, y_pred, y_true in zip(final_prob, final_pred, y_test):
        if y_true == 0 and y_true != y_pred:
          misclassified_probs_nc.append(prob)
        if y_true == 1 and y_true != y_pred:
          misclassified_probs_c.append(prob)
      
      sns.set_theme(style="whitegrid")
      
      ax_1 = sns.violinplot(x=misclassified_probs_nc)
      # plt.clf()
      # ax_2 = sns.violinplot(x=misclassified_probs_c)
    
    if print_preds:
      print("final predictions:", list(final_pred))
      print("true labels:", y_test)

P_INIT = True

RUN_ID = 1
VOTING_ONLY = False
EMBEDDING_TYPE = None # try: 'glove'
BATCH_SIZE = 32
model_prefix = '_claim_lstm_orthogonalproj_def_focal_new'
if __name__ == "__main__":
  if RUN_ID == 1:
    # DATASET CLEANING AND DEPENDENCY PARSING (OPTIONAL)
    dataset = pd.read_csv('_submission_data_raw.csv', encoding = "utf-8")
    _dataset_obj = Dataset(dataset, 'tweet_text')
    dataset = _dataset_obj.preprocess(p_init = P_INIT, get_dep=False)
    print("DATASET CLEANSED")
    print("+--------------------------------------+\n")

    # FETCH POINCARE EMBEDDING
    wvpoincare = WVPoincare(filename='sentiment160_poincare_512_1.pickle')
    print("FETCHED POINCARE EMBEDDING")
    print("+--------------------------------------+\n")

    # FORMULATE TEXT DATA
    _tdl = TextDataLoader(
        dataset=dataset,
        text_column='tweet_text',
        label_column='claim',
        embedding_model=wvpoincare.model,
        num_words=30000
    )

    _tdl.get_embeddingsindex(filename="sentiment160_poincare_512_1.txt", embedding_type = EMBEDDING_TYPE)
    tokenizer = _tdl.get_tokenizer(return_tokenizer=True)
    vocab_size, embedding_matrix = _tdl.get_embeddingmatrix(return_embeddingmatrix=True)
    texts, labels = _tdl.get_data()
    print("TOKENIZER AND EMBEDDING MATRIX FORMULATED")
    print("+--------------------------------------+\n")

  _tdg = TextDataGenerator(
      texts=texts,
      labels=labels,
      tokenizer=tokenizer,
      sampling_strategy=0.40
  )

  # DEFINITIONS
  definitions = {
      'claim': [
                'texts mentioning statistics, dates or numbers',
                'texts mentioning a personal experience',
                'texts reporting something to be true or reporting an occured instance',
                'texts containing verified facts account for a claim',
                'texts that negate a possibly false claim account for a claim',
                'texts that indirectly imply that something is true',
                'claims made in sarcasm or humor',
                'opinions with societal implications',
                'texts that say something is true with evidence',
                'claims can be a sub part of a question'
      ],
      'non-claim': [
                    'phrases with i guess or i suppose',
                    'hoping that something happens is not claiming it',
                    'inclusion of words that project doubt over the said statement',
                    'urging one to not claim something or to spread misinformation is not a claim',
                    'questioning a possible claim is not a claim',
                    'warning someone against a claim is not a claim',
                    'ouright assuming something, is not claiming it',
                    'miscellaneous statements'
      ]
  }
  definitions['claim'] = _tdg.text_encoder(definitions['claim'])
  definitions['non-claim'] = _tdg.text_encoder(definitions['non-claim'])

  # CALIBRATION
  calibrate = {
      "auto": False,
      "manual": 0.45
  }

  _num_folds = 5
  for fold in range(_num_folds):
    train_generator, validation_generator, test_generator = _tdg.get_generator(fold_id = fold)
    print("FOLD {}:".format(fold))

    # INITIALIZE AND TRAIN MODEL
    generators = {
      'train': train_generator,
      'validation': validation_generator,
      'test': test_generator
      }

    _model = MyModel(
        definitions=definitions,
        vocab_size=vocab_size,
        EMBEDDING_DIM=100,
        embedding_matrix=embedding_matrix,
        maxlen=50,
        lstm_hidden_units = 256,
        batch_size = BATCH_SIZE
    )
    _pnet = _model.lstm_network(model_id=1, induce_definition=True)
    _cnet = _model.lstm_network(model_id=2, induce_definition=False)

    pnet = {
        "model": _pnet,
        "layer_name": "bidirectional_2.1"
    }
    cnet = {
        "model": _cnet,
        "layer_name": "bidirectional_2.2"
    }

    _model.featureproj_model(pnet=pnet, cnet=cnet, custom_loss='focal')
    
    if VOTING_ONLY:
      _model.voting_classifier(generators=generators, calibrate=None, error_analysis=False, print_preds=True, model_prefix=model_prefix)
      break
    
    _model.train(generators=generators, fold_id=fold, epochs=100, batch_size=BATCH_SIZE, model_prefix=model_prefix, return_model=False, cw=False, cp=False, es=True, ets_restore=True)

    _model.evaluate()
    print("________________________\n")
    
    if fold == 4:
      _model.voting_classifier(model_prefix=model_prefix)
  
  print("TRAINED AND EVALUATED")
  print("+--------------------------------------+\n")


'''
# RUN AFTER COMPUTING Y_HAT FOR THE 3 DIFFERENT VALUES OF GAMMA IN FOCAL LOSS
# R_INIT
y_true = [
  # insert values
]
focal_1 = [
  # insert values
]
focal_2 = [
  # insert values
]
focal_3 = [
  # insert values
]
print(classification_report(y_true, most_common([focal_1, focal_2, focal_3]), target_names=['non-claim', 'claim']))
'''