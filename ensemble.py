import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import bz2
import gc
import chardet
import re
import os
from keras.preprocessing import text, sequence
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Input, Conv1D, Flatten
from keras.layers import Dropout, Bidirectional, CuDNNLSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.utils.conv_utils import conv_output_length
from keras.regularizers import l2
from keras.constraints import maxnorm
from custom_layers import Attention, Capsule
from sklearn.linear_model import LogisticRegression


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

max_features = 20000
maxlen = 100

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train.sentences)

tokenized_train = tokenizer.texts_to_sequences(train.sentences)
X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)

tokenized_test = tokenizer.texts_to_sequences(test.sentences)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)


EMBEDDING_FILE = 'data/glove.6B.50d.txt'

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
#change below line if computing normal stats is too slow
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size)) #embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector



batch_size = 2048
epochs = 20
embed_size = 50





def capsule_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Dropout(0.25)(x)
    x = Conv1D(2*embed_size, kernel_size = 3)(x)
    x = Bidirectional(CuDNNLSTM(128, kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6), 
               kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10), return_sequences=True))(x)
    print(x.shape)

    x = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x)
    x = Flatten()(x)

    x = Dropout(0.5)(x)
    x = Dense(22, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])

    return model


def attention_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size,
                  weights=[embedding_matrix], trainable=False)(inp)
    x = Dropout(0.25)(x)
    x = Conv1D(2*embed_size, kernel_size=3)(x)
    x = Bidirectional(CuDNNLSTM(128, kernel_regularizer=l2(4e-6),
                      bias_regularizer=l2(4e-6), kernel_constraint=maxnorm(10),
                      bias_constraint=maxnorm(10), return_sequences=True))(x)
    x = Attention(x.shape[1])(x)
    x = Dropout(0.5)(x)
    x = Dense(22, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])

    return model

def cudnnlstm_model(conv_layers = 2, max_dilation_rate = 3):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = Dropout(0.25)(x)
    x = Conv1D(2*embed_size, kernel_size = 3)(x)
    x = Conv1D(2*embed_size, kernel_size = 3)(x)
    for strides in [1, 1, 2]:
        x = Conv1D(128*2**(strides), strides = strides, kernel_regularizer=l2(4e-6), 
                   bias_regularizer=l2(4e-6), kernel_size=3, kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10))(x)
    x = Bidirectional(CuDNNLSTM(256, kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6), 
                                kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))(x)  
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])

    return model


attention_clf = attention_model()
capsule_clf = capsule_model()
bilstm_clf = cudnnlstm_model()

attention_clf.load_weights('attention_weights.hdf5')
capsule_clf.load_weights('capsule_weights.hdf5')
bilstm_clf.load_weights('bilstm_weights.hdf5')

attention_preds = attention_clf.predict(X_test).reshape(-1)
capsule_preds = capsule_clf.predict(X_test).reshape(-1)
bilstm_preds = bilstm_clf.predict(X_test).reshape(-1)

naive_ensemble_preds = (attention_preds + capsule_preds + bilstm_preds)/3

# linear model ensemble
attention_preds_train = attention_clf.predict(X_train[:500_000])
capsule_preds_train = capsule_clf.predict(X_train[:500_000])
bilstm_preds_train = bilstm_clf.predict(X_train[:500_000])
all_preds_train = np.hstack([
	attention_preds_train,
	capsule_preds_train,
	bilstm_preds_train])
all_preds_test = np.hstack([
	attention_preds.reshape(-1, 1),
	capsule_preds.reshape(-1, 1),
	bilstm_preds.reshape(-1, 1),
	])
lr = LogisticRegression()
lr.fit(all_preds_train, train.labels.head(500_000))
lr_ensemble_preds = lr.predict(all_preds_test)

bilstm_score = ((bilstm_preds > 0.5) == test.labels).mean()
capsule_score = ((capsule_preds > 0.5) == test.labels).mean()
attention_score = ((attention_preds > 0.5) == test.labels).mean()
naive_ensemble_score = ((naive_ensemble_preds > 0.5) == test.labels).mean()
lr_ensemble_score = ((lr_ensemble_preds > 0.5) == test.labels).mean()

print(f'BiLSTM binary accuracy: {round(bilstm_score, 3)}')
print(f'Attention binary accuracy: {round(attention_score, 3)}')
print(f'Capsule binary accuracy: {round(capsule_score, 3)}')
print(f'Naive Ensemble binary accuracy: {round(naive_ensemble_score, 3)}')
print(f'Linear model Ensemble binary accuracy: {round(lr_ensemble_score, 3)}')

