import numpy as np
import random
import keras
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, Masking, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import callbacks
import csv
import tensorflow as tf
import os
import random
from tensorflow.python.keras.callbacks import TensorBoard 
from time import time

from tensorboard_logging import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

best_dense = [
    [23, 83, 228, 263, 163, 241],
    [20, 107, 267, 271, 295],
    [19, 281, 270, 143, 54, 239, 194],
    [20, 251, 290, 140, 59, 257, 187],
    [25, 131, 257, 267, 281],
    ]

epochs = 500
n_joints = 10
n_cv = 5
batch_size = 32

to_train = 3000
n_best = 0

bestmodel = None
bestperformance = None

from BachelorUtilities import loadData
eul, crp, das28 = loadData('Training')

def getData(eul, crp, das28, shuffle=False):
    # Generate the permutation index array.
    permutation = np.random.permutation(eul.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.

    result_eul = np.array_split(np.reshape(eul, (-1, n_joints, 1)), n_cv)
    result_crp = np.array_split(crp, n_cv)
    result_das28 = np.array_split(das28, n_cv)

    return result_eul, result_crp, result_das28

def crossValidation(data, i):
    result = []
    for d in data:
        dT = np.array([])
        for x in range(0, len(d)):
            if x is not i:
                if len(dT) is 0:
                    dT = d[x]
                else:
                    dT = np.append(dT, d[x], 0)
            else:
                dV = d[x]
        result.append([dT, dV])
    return result

x1, x2, y = getData(eul, crp, das28)
crossValidations = []
for i in range(0, n_cv): crossValidations.append(crossValidation([x1, x2, y], i))

for i, dataset in enumerate(crossValidations):
    [[x1t, x1v], [x2t, x2v], [yt, yv]] = dataset
    m = np.mean(x2t)
    std = np.std(x2t)
    x2t = (x2t - m) / std
    x2v = (x2v - m) / std
    crossValidations[i] = [[x1t, x1v], [x2t, x2v], [yt, yv]]

for n_model in range(0, to_train):
    # INITIALIZERS
    dense_nodes = [281, 270, 143, 54, 239, 194, 168]
    lstm_size = 19
    learning_rate = 0.0005
    dropout = 0.1

    seed = None

    dense_initializer = keras.initializers.RandomNormal(seed=seed)
    lstm_kernel_initializer = keras.initializers.glorot_uniform(seed=seed)
    lstm_recurrent_initializer = keras.initializers.orthogonal(seed=seed)

    histories = []
    histories_mae = []
    models = [None] * n_cv

    weights = None

    for i, [[x1t, x1v], [x2t, x2v], [yt, yv]] in enumerate(crossValidations):
        keras.backend.clear_session()
        # CALLBACKS
        cb_tensorboard = keras.callbacks.TensorBoard(log_dir='D:\WindowsFolders\Documents\TensorboardLogs\{}_{}'.format(lstm_size,i), histogram_freq=0, write_graph=True, write_images=True)
        cb_tensorboard_model = keras.callbacks.TensorBoard(log_dir='D:\WindowsFolders\Documents\TensorboardLogs\{}_{}'.format(lstm_size,i), histogram_freq=0, write_graph=False, write_images=True)
        cb_earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=15, verbose=False, restore_best_weights=True)
        cb_lrsched = callbacks.ReduceLROnPlateau(monitor='val_loss',verbose=False,patience=5)
        cbs = [cb_earlystop, cb_lrsched]

        # Input definition
        input_eular = Input(shape=(n_joints, 1), dtype='float32', name='input_eular')
        input_crp = Input(shape=(1,), dtype='float32', name='input_crp')

        # Model definition
        x = Masking(mask_value=-1)(input_eular)
        x = LSTM(lstm_size, return_sequences=False, kernel_initializer=lstm_kernel_initializer, recurrent_initializer=lstm_recurrent_initializer)(x)
        x = concatenate([x, input_crp])
        for j, nodes in enumerate(dense_nodes): 
            if j is not 0:
                x = Dropout(rate=dropout)(x)
            x = Dense(nodes, activation='relu', kernel_initializer=dense_initializer)(x)
        output = Dense(1, kernel_initializer=dense_initializer)(x)

        model = Model(inputs=[input_eular, input_crp], outputs=output)

        optimizer = Adam(lr=learning_rate)

        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        if i == 0:
            model.save_weights('weights/model_weights_{}'.format(n_model,i))
        else:
            model.load_weights('weights/model_weights_{}'.format(n_model,i))

        hist = model.fit([x1t, x2t],
            yt,
            validation_data=([x1v, x2v], yv),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cbs,
            verbose=False)
        
        keras.models.save_model(model, 'model_{}.h5'.format(i))

        histories.append(hist.history['val_loss'])
        histories_mae.append(hist.history['val_mean_absolute_error'])

    N = len(histories)
    M = max([len(h) for h in histories])

    avg = [0] * M

    for i in range(0, N):
        for j in range (0, M):
            if len(histories[i]) <= j:
                avg[j] += histories[i][-1]
            else:
                avg[j] += histories[i][j]
                
    avg_mae = [0] * M

    for i in range(0, N):
        for j in range (0, M):
            if len(histories_mae[i]) <= j:
                avg_mae[j] += histories_mae[i][-1]
            else:
                avg_mae[j] += histories_mae[i][j]

    if bestperformance is None or avg[-1] < bestperformance:

        bestperformance = avg[-1]
            
        with open('results.csv', mode='a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow([n_model, dense_nodes, lstm_size, learning_rate, dropout, avg_mae[-1]/N, avg[-1]/N])

        print('New best model. Last mse:', avg[-1]/N, 'Last mae:', avg_mae[-1]/N, '')
        n_best += 1

    else:     
        print('Done with model',n_model,'of',to_train,'.',avg_mae[-1]/N)


