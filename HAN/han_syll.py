

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("/content/IIITH_Codemixed_new.txt",sep="\t",names = ["Speech", "Labels"]).dropna()
df1 = pd.read_csv("/content/train_new.txt",sep = "\t",names = ["Speech", "Labels"]).dropna()
df_new = pd.concat([df,df1], ignore_index=True)
df = df_new
df["Labels"] = df["Labels"].astype("int")
df.head()

label_counts = df["Labels"].value_counts().tolist()
labels = df["Labels"].value_counts().index.tolist()
labels = list(map(int, labels))
print(label_counts)

plt.bar(labels, label_counts)

#Data Cleaning
#Removing all the punctuation marks and converting to lowercase
import re

def clean_str(string):
    """Tokenization/string cleaning for all datasets except for SST.

    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"[.,#!$%&;:{}=_`~()/\\]", "", string)
    return string.strip().lower()

df['Speech'] = df['Speech'].astype(str).map(clean_str,na_action=None)


df = df.sample(frac=1).reset_index(drop=True)
df.head()

labels = df["Labels"].values

text = df['Speech']

sentences1 = text.apply(lambda x: x.split())
#storing the original dataset without syllable in an object
sent_without_syllable = sentences1

#dataset without syllable level decomposition
sent_without_syllable.head()

import numpy as np
#Creating the syllable decomposition
def ortho_syllable(word):
    """Split word to orhtographic syllable."""
    vector = vectorize(word)
    grad_vec = gradient(vector)
    SW = ""
    i = 0
    w_len = len(word)
    while(i < w_len):
        SW = SW + word[i]
        if (i+1) < w_len:
            if i == 0 and grad_vec[i] == -1:
                SW = SW + word[i+1] + " "
                i += 1
            elif grad_vec[i] == -1 and i != w_len-1:
                if word[i+1] in ['r', 's', 't', 'l', 'n', 'd'] and i+1 != w_len-1:
                    if vector[i+2] == 0:
                        SW = SW + word[i+1]
                        i += 1
                SW = SW + " "
        i += 1
    # pdb.set_trace()
    return SW.split()


def is_vowel(char):
    """Check if it is vowel."""
    return char in ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']


def gradient(vector):
    """Get the gradient of the vector."""
    vec2 = vector[1::]
    vec2.append(0)
    vec2 = np.array(vec2)
    vec = np.array(vector)
    return vec2-vec


def vectorize(word):
    """Vectorize based on consonant and vowel."""
    vec = list()
    for i in range(len(word)):
        vec.append(int(is_vowel(word[i])))
    return vec


import csv
def syllable(arr):
  with open("/content/code_mixed_syllable.txt", "w") as f:
    writer = csv.writer(f, delimiter=" ")
    for i in range(arr.shape[0]):
      sent = arr[i]
      new_sen = []
      for word in sent:
        word = ortho_syllable(word)
        for syll in word:
          new_sen.append(syll)
      f.write(','.join(str(i) for i in new_sen)) 
      f.write('\n')
      
syllable(sent_without_syllable.values)

#Loading the saved text file into a dataframe
#this text file contains syllable level decomposition for the words
df2 = pd.read_csv("/content/code_mixed_syllable.txt", sep = " ", names=["speech"])
df2 = df2.applymap(str)
df2.head()

#Dataset with syllables
text1 = df2["speech"]
sen_with_syllable = text1.apply(lambda x: x.split(","))
sen_with_syllable.head()

#imports for the models
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed,Dropout,SpatialDropout1D
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam

#Original code taken from https://www.kaggle.com/sermakarevich/hierarchical-attention-network

class AttentionWithContext(Layer):
    def __init__(self, attention_dim,**kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttentionWithContext, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = K.variable(self.init((self.attention_dim, )), name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
        self._trainable_weights = [self.W, self.b, self.u]
        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
      
      config = super().get_config().copy()
      config.update({
              'attention_dim': self.attention_dim })
      return config

MAX_WORD_LENGTH = 15  # max length of the sentence to _ no of chars
MAX_WORDS = 25 # Max no of words in a given sentence
MAX_syll = 10000  # how many unique sylls to use ( i.e num rows in embedding vector)
EMBEDDING_DIM = 25   # how big is each word vector
VALIDATION_SPLIT = 0.2

#Use the below objects for tokenizing 

#Tokenizer for sentences without syllable


tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(sent_without_syllable)



#Tokenizer for sentences with syllable
tokenizer_syllable = Tokenizer(num_words=MAX_WORDS, char_level=False, oov_token="<OOV>")
tokenizer_syllable.fit_on_texts(sen_with_syllable)

def prepare_data(sentences_input):

  data = np.zeros((len(sentences_input), MAX_WORDS, MAX_WORD_LENGTH), dtype='int32')

  for i, words in enumerate(sentences_input):
      for j, word in enumerate(words):

        word = ortho_syllable(word)
        if j < MAX_WORDS:
            k = 0
            for _, syll in enumerate(word):
                try:
                    if k < MAX_WORD_LENGTH:
                        if tokenizer_syllable.word_index[syll] < MAX_syll:
                            data[i, j, k] = tokenizer_syllable.word_index[syll]
                            k=k+1
                except:
                    None
  return data 

input_data = prepare_data(sen_with_syllable)

from sklearn import preprocessing
import keras.utils.np_utils

encoder = preprocessing.LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = keras.utils.np_utils.to_categorical(encoded_Y)
labels = dummy_y
labels

import keras.utils.np_utils
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(input_data, labels, test_size = 0.2)


indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]


nb_validation_samples = int(VALIDATION_SPLIT * X_train.shape[0])

x_train = X_train[:-nb_validation_samples]
y_train = Y_train[:-nb_validation_samples]
x_val = X_train[-nb_validation_samples:]
y_val = Y_train[-nb_validation_samples:]
print(x_train[0].shape)
print('Number of positive and negative and neutral reviews in training and validation set')
print (y_train.sum(axis=0))
print (y_val.sum(axis=0))

char_index = tokenizer_syllable.word_index
print('Total %s unique tokens.' % len(char_index))

import tensorflow as tf
reg = 1e-13

syll_input = Input(shape=(MAX_WORD_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(char_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_WORD_LENGTH,
                            trainable=False,name='word_embedding')(syll_input)

#syll_input = Input(shape=(MAX_WORD_LENGTH,), dtype='int32')
#char_sequences = embedding_layer(syll_input)

char_lstm = Bidirectional(LSTM(50, return_sequences=True))(embedding_layer)

char_att = AttentionWithContext(50)(char_lstm)
charEncoder = Model(syll_input, char_att)

words_input = Input(shape=(MAX_WORDS, MAX_WORD_LENGTH), dtype='int32')
words_encoder = TimeDistributed(charEncoder)(words_input)
#words_encoder = SpatialDropout1D(0.2)(words_encoder)
words_lstm = Bidirectional(LSTM(50, return_sequences=True))(words_encoder)
#words_dense = TimeDistributed(Dense(200))(words_lstm)
words_att = AttentionWithContext(50)(words_lstm)
word_drop = Dropout(0.5,name='words_dropout')(words_att)
preds = Dense(3, activation='softmax')(words_att)
model_han = Model(words_input, preds)

model_han.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print(model_han.summary())

"""### Training"""

TF_FORCE_GPU_ALLOW_GROWTH=True
#RocAuc = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)
history = model_han.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=40, batch_size=64)

#testing the model
from matplotlib import pyplot
# evaluate the model
_, train_acc = model_han.evaluate(x_train, y_train, verbose=0)
_, test_acc = model_han.evaluate(X_test, Y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validate')
pyplot.legend()
pyplot.show()









