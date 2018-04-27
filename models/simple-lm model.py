# overview of plan:
# tokenize text
# create embedding
# run through LSTM layers
# predict embedding of next word
# loss: cosine distance between true embedding value and predicted
# 
# For generating samples, start with random word(s) from the corpus? <SOT> token?
# generate predictions according to fun calc
# choose random next word according to probs (initially - just use closest embedded word)
# NOTE: this has a very simple failure state - if the embedding becomes constant then there is no loss
# use GloVe to train word embeddings on the entire document
# we will proceed by assuming we have done this, and have the trained embeddings

from keras.layers import Input, Dense, LSTM, Masking, Activation, Dropout
from keras.model import Model
from keras.utils import to_categorical
from keras.preprocessing import Tokenizer, one_hot
import pandas as pd
import numpy as np
from scipy.spatial import cdist

[[load data]]

number_of_words = 1000

text_data = Tokenizer(data, oov_token = '<UNK>')

[[generate imputs]]



class simple-generator(object):
    def __init__(self):
        self.embedding_index = dict()
        with open('../data/glove.bible.100d.txt') as f:
            for line in f:
                values = line.split()
                word=values[0]
                coefs = asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
        # also build dictionary to translate words to indicies
        # then use
        inputs = Input(shape=(None,1))
        x = Embedding(1, dimensions, weights=[embedding_matrix], trainable=False)(x)
        x = Masking(x)
        x = LSTM(num_nodes, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(num_nodes, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        predictions = TimeDistributed(Dense(dimensions, activation='softmax')(x))
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


    def generate(self, eps=1):
        # start with BOT marker / provided word
        # feed into model
        # take prediction vector
        # compare to all vectors in dictionary, using cosine similarity
        # SIMPLE: take closest
        # COMPLEX: take distance vector
        #   calculate normal prob of at least that distance (2*SF(X/eps))
        #   normalize this vector
        #   use as prob vector to select entry at random
        # input chosen symbol into generator
        # repeat until EOT marker, timeout, or too many iterations (80 should do it - longest text is 46)
        # 
        
        dists = cdist(predicted, embedding_weights)
        probs = np.reshape(2*scipy.stats.norm.sf(dists, shape=eps), (dimensions))
        pred_token = np.random.choice(dimensions, p=probs)

