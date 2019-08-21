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
from keras.layers import Dense, Embedding, Input, Conv1D
from keras.layers import Dropout, Bidirectional, CuDNNLSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.utils.conv_utils import conv_output_length
from keras.regularizers import l2
from keras.constraints import maxnorm


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
EMBEDDING_FILE = f'{data_folder}/glove.6B.50d.txt'

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

model = cudnnlstm_model()
print(model.summary())


weight_path="bilstm_weights.hdf5"
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)
callbacks = [checkpoint, early_stopping]

# model.fit(X_train, train.labels, batch_size=batch_size, epochs=epochs, shuffle = True, validation_split=0.20, callbacks=callbacks)

model.load_weights(weight_path)
test_predictions = model.predict(X_test)
score, acc = model.evaluate(X_test, test_labels, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

predictions = model.predict(X_test)
predictions_df = pandas.DataFrame({'bilstm_preds': model.predict(X_test)})
predictions_df.to_csv('bilstm_preds.csv')
