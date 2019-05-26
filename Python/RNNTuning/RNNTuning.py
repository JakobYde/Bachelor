import numpy as np
import random
import keras
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, Masking, Dropout
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

max_layers = 7

n_samples = 5000

# Random Search Parameters
n_cv = 5

# Variation
size_parameter_pool = 500
size_survivors = 10
size_original = 10
window_size = 5


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


def loadParameters(filename):
    parameter_pool = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)

        for i, row in enumerate(reader):
            n = int(row[1])
            if row[0] is "":
                learning_rate = 0.001
            else:
                learning_rate = float(row[0])
            layers = []
            for j in range(0, n): layers.append(int(row[j + 2]))

            parameters['learning_rate'] = learning_rate
            parameters['n_layers'] = n
            parameters['dense_nodes'] = layers

            parameter_pool.append(parameters)

    return parameter_pool


def writeCsvHead(col_names):
    with open('results.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(col_names)


def augmentParameters(parameters, n_params):
    result = []
    params = []

    amount_of_params = 0
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    layer_changes = [-20, -10, -5, -2, -1, 0, 0, 0, 1, 2, 5, 10, 20]
    div = 0
    for p in parameters: 
        if p != {}:
            div += 1
    for parameter in parameters:
        if parameter != {}:
            params = [parameter]
            lackingParams = int(n_params / div - len(params))
            while lackingParams != 0:
                newParams = []
                for i in range (0, lackingParams):
                    param = {}       
                    param['learning_rate'] = random.choice(learning_rates)
                    param['n_layers'] = parameter['n_layers']
                    param['dense_nodes'] = []
                    param['lstm'] = -1

                    for i, n in enumerate(parameter['dense_nodes']):
                        newn = 0
                        while newn <= 0:
                            newn = n + random.choice(layer_changes)
                        param['dense_nodes'].append(newn)
                    if param not in params and param not in newParams:
                        newParams.append(param)

                    while param['lstm'] <= 0: param['lstm'] = parameter['lstm'] + random.choice(layer_changes)
                if len(newParams) > 0: params += newParams
                            
                lackingParams = int(n_params / div - len(params))
            result += params

    return result


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


def trainNetwork(parameters, x1t, x1v, x2t, x2v, yt, yv):
    # Hyperparameters
    learning_rate = parameters['learning_rate']
    output_lstm = parameters['lstm']
    layers = parameters['n_layers']
    dense_nodes = parameters['dense_nodes']
    print_layers = [0] * max_layers
    for i, v in enumerate(dense_nodes): print_layers[i] = v
    validation_size = x1v.shape[0]

    dense_initializer = keras.initializers.RandomNormal(seed=7051)
    lstm_kernel_initializer = keras.initializers.glorot_uniform(seed=7051)
    lstm_recurrent_initializer = keras.initializers.orthogonal(seed=7051)

    # Input definition
    input_eular = Input(shape=(n_joints, 1), dtype = 'float32', name='input_eular')
    input_crp = Input(shape=(1,), dtype='float32', name='input_crp')

    # Model definition
    x = Masking(mask_value=-1)(input_eular)
    x = LSTM(output_lstm, return_sequences=False, kernel_initializer=lstm_kernel_initializer, recurrent_initializer=lstm_recurrent_initializer)(x)
    x = concatenate([x, input_crp])
    for nodes in dense_nodes: x = Dense(nodes, activation='relu', kernel_initializer=dense_initializer)(x)
    output = Dense(1)(x)

    model = Model(input=[input_eular, input_crp], output=output)

    optimizer = Adam(lr=learning_rate)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    hist = model.fit([x1t, x2t],
                     yt,
                     validation_data=([x1v, x2v], yv),
                     epochs=epochs,
                     batch_size=batch_size,
                     verbose=False)

    head = [learning_rate]
    for l in print_layers: head.append(l)
    head.append(validation_size)
    mae = hist.history['val_mean_absolute_error']
    mse = hist.history['val_loss']
    t_mse = hist.history['loss']

    print_list = []
    print_list += head
    print_list.append(':')
    print_list.append('val_mae')
    print_list += mae
    print_list.append(':')
    print_list.append('val_mse')
    print_list += mse
    print_list.append(':')
    print_list.append('train_mse')
    print_list += t_mse

    print("{:.2f}".format(np.min(mse)), "-", head)

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
    for i in range(0, len(performance) - window_size + 1): filtered_performance.append(np.mean(performance[i:i + window_size]))

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

# params = getRandomStructure(20000, [range_lstm, range_layers, range_nodes, range_learning_rate], ['lstm', 'n_layers', 'dense_nodes', 'learning_rate'])
# np.save("Models.npy", np.array(stru))

# Data
eul = np.load("TrainingDataEUL.npy")
eul = np.array(eul, dtype='float32')
crp = np.load("TrainingDataCRP.npy")
das28 = np.load("TrainingDataY.npy")

writeCsvHead(['learning_rate', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7', 'val_size'])

inner_size = [{} for i in range(size_survivors)]
original_parameters = [np.copy(inner_size) for i in range(size_original)]

with open('inc_bestmodels.csv', mode='r') as csv_file:
    models = csv.reader(csv_file)
    for i, row in enumerate(models):
        params = {}
        params['learning_rate'] = float(row[0])
        params['lstm'] = int(row[2])
        params['n_layers'] = int(row[1])
        params['dense_nodes'] = [int(j) for j in row[3:3+int(row[1])]]
        while 0 in params['dense_nodes']: params['dense_nodes'].remove(0)
        original_parameters[i][0] = params

original_parameters = original_parameters[2:8]

#################
## ENTRY POINT ##
#################

generation = 1
min_loss = 100

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

for k in range(0,2):
    for i, parameter in enumerate(original_parameters):
        parameter_pool = augmentParameters(parameter, size_parameter_pool)
        parameter_performance = {}
        np.save('allgens/parameter_pool', parameter_pool)
        for j, parameters in enumerate(parameter_pool):
            performances = []
            for k in range(n_cv):
                [[x1t, x1v],[x2t, x2v],[yt, yv]] = crossValidation([x1, x2, y], k)
                performances.append(trainNetwork(parameters, x1t, x1v, x2t, x2v, yt, yv))
            new_loss = getPerformance(performances, window_size)
            parameter_performance[new_loss] = parameters
            if new_loss < min_loss:
                min_loss = new_loss
                print("New min loss:",new_loss,"Parameters:",parameters)
            np.save('allgens/j.npy', np.array([j]))
        best_keys = sorted(parameter_performance.keys())[:size_survivors]
        new_parameters = [parameter_performance[x] for x in best_keys]
        original_parameters[i] = new_parameters
        np.save('allgens/paramsgen{}_{}.npy'.format(generation, i), original_parameters)
        print("parameter done.")
    print("Generation", generation, "done.")
    generation += 1