import numpy as np
import keras
from keras.layers import Input, Embedding, Dense, concatenate, Masking, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
import csv
import tensorflow as tf
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Constants
n_joints = 10
epochs = 100
batch_size = 128
validation_split = 0.15

max_layers = 7

n_samples = 5000

LEARNING_RATE = 0
LAYERS = 2
NODES = 3

# Random Search Parameters
n_cv = 5

def getData(eul, crp, das28, shuffle=False):
    # Generate the permutation index array.
    permutation = np.random.permutation(eul.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    if shuffle:
        shuffled_eul = eul[permutation]
        shuffled_crp = crp[permutation]
        shuffled_das28 = das28[permutation]
    else:
        shuffled_eul = eul
        shuffled_crp = crp
        shuffled_das28 = das28

    result_eul = np.array_split(np.reshape(shuffled_eul, (-1, n_joints, 1)), n_cv)
    result_crp = np.array_split(shuffled_crp, n_cv)
    result_das28 = np.array_split(shuffled_das28, n_cv)

    return result_eul, result_crp, result_das28



def writeCsvHead(col_names):
    with open('results.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(col_names)


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

def trainNetwork(parameters, x1t, x1v, x2t, x2v, yt, yv, use_same_model):

    # Categorize eular-omeract scores

    x1t = keras.utils.to_categorical(x1t, 5)
    x1v = keras.utils.to_categorical(x1v, 5)

    x1t = np.array([i.flatten() for i in x1t])
    x1v = np.array([i.flatten() for i in x1v])

    # Hyperparameters
    learning_rate = parameters['learning_rate']
    layers = parameters['n_layers']
    dense_nodes = parameters['dense_nodes']
    print_layers = [0] * max_layers
    for i, v in enumerate(dense_nodes): print_layers[i] = v
    validation_size = x1v.shape[0]

    # Reshape input
    x1t = np.reshape(x1t, (x1t.shape[0], x1t.shape[1]))
    x1v = np.reshape(x1v, (x1v.shape[0], x1v.shape[1]))

    # Input definition
    input_eular = Input(shape=(50,), dtype='float32', name='input_eular')
    input_crp = Input(shape=(1,), dtype='float32', name='input_crp')

    # Model definition
    x = concatenate([input_eular, input_crp], axis=1)
    for nodes in dense_nodes: x = Dense(nodes, activation='relu')(x)
    output = Dense(1)(x)

    model = Model(input=[input_eular, input_crp], output=output)

    optimizer = Adam(lr=learning_rate)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    cb_lrs = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=40, verbose=0)
    cbs = [cb_lrs]

    if not use_same_model:
        model.save_weights('model_weights.h5')

    if use_same_model:
        model.load_weights('model_weights.h5')

    hist = model.fit([x1t, x2t],
                     yt,
                     validation_data=([x1v, x2v], yv),
                     epochs=epochs,
                     batch_size=batch_size,
                     verbose=False, callbacks=cbs)

    mae = hist.history['val_mean_absolute_error']
    mse = hist.history['val_loss']
    t_mse = hist.history['loss']

    print_list = [learning_rate]
    print_list.append(validation_size)
    print_list.append(':')
    print_list.append('val_mae')
    print_list += mae
    print_list.append(':')
    print_list.append('val_mse')
    print_list += mse
    print_list.append(':')
    print_list.append('train_mse')
    print_list += t_mse

    with open('results.csv', mode='a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(print_list)

    keras.backend.clear_session()
    return mse

def getPerformance(performances, window_size):
    performance = []
    # Averaging across cross validation.
    for i in range(0, len(performances[0])): performance.append(np.mean([d[i] for d in performances]))

    filtered_performance = []
    d = int(window_size / 2)
    # Low-pass filtering result.
    for i in range(0, len(performance) - window_size + 1): filtered_performance.append(
        np.mean(performance[i:i + window_size]))

    # return min value
    return np.min(filtered_performance)

def getRandomStructure(N, param_ranges, param_names):
    result = []
    n_param = len(param_ranges)

    while len(result) < N:
        parameter_set = {}
        parameter_set[param_names[0]] = random.choice(param_ranges[0])  # LSTM NODES
        n_layers = random.choice(param_ranges[1])  # LAYERS
        parameter_set[param_names[1]] = n_layers
        layers = []
        for _ in range(0, n_layers):
            layers.append(random.choice(param_ranges[2]))
        parameter_set[param_names[2]] = layers
        parameter_set[param_names[3]] = random.choice(param_ranges[3])
        if parameter_set not in result:
            result.append(parameter_set)

    return result

window_size = 5
min_loss = 100

# Data
eul = np.load("TrainingDataEUL.npy")
eul = np.array(eul, dtype='float32')
crp = np.load("TrainingDataCRP.npy")
das28 = np.load("TrainingDataY.npy")

params_list = [{'learning_rate': 0.005, 'n_layers': 6, 'dense_nodes': [134, 189, 157, 220, 201, 235]}]

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

writeCsvHead(['learning_rate', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7', 'val_size'])

learning_rates = 0.005

amount_of_runs = 50

for params in params_list:
    use_same_model = False
    for s in range(amount_of_runs):
        performances = []
        for i in range(0, n_cv):
            [[x1t, x1v], [x2t, x2v], [yt, yv]] = crossValidations[i]
            performances.append(trainNetwork(params, x1t, x1v, x2t, x2v, yt, yv, use_same_model=use_same_model))
            use_same_model = True
        new_loss = getPerformance(performances, window_size)
        if new_loss < min_loss:
            min_loss = new_loss
            print("New min loss:", new_loss, "Parameters:", model)
        print(s)
