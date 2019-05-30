import numpy as np
import random
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

                    for i, n in enumerate(parameter['dense_nodes']):
                        newn = 0
                        while newn <= 0:
                            newn = n + random.choice(layer_changes)
                        param['dense_nodes'].append(newn)
                    if param not in params and param not in newParams:
                        newParams.append(param)
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
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=7051)
    x = concatenate([input_eular, input_crp], axis=1)
    for nodes in dense_nodes: x = Dense(nodes,  kernel_initializer=initializer, activation='relu')(x)
    output = Dense(1,  kernel_initializer=initializer)(x)

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
    for i in range(0, len(performance) - window_size + 1): filtered_performance.append(
        np.mean(performance[i:i + window_size]))

    # return min value
    return np.min(filtered_performance)

#################
## ENTRY POINT ##
#################

generation = 1
window_size = 5
generation_size = 10
min_loss = 100

# Data
eul = np.load("TrainingDataEUL.npy")
eul = np.array(eul, dtype='float32')
crp = np.load("TrainingDataCRP.npy")
das28 = np.load("TrainingDataY.npy")

# Create array containing the best models from the previous test
# These will be replaced in the while True loop after each generation for each model.
inner_size = [{} for i in range(generation_size)]
original_parameters = [np.copy(inner_size) for i in range(10)]

with open('bestmodels.csv', mode='r') as csv_file:
    models = csv.reader(csv_file)
    for i, row in enumerate(models):
        params = {}
        params['learning_rate'] = float(row[0])
        params['n_layers'] = int(row[1])
        params['dense_nodes'] = [int(j) for j in row[2:]]
        while 0 in params['dense_nodes']: params['dense_nodes'].remove(0)
        original_parameters[i][0] = params
original_parameters = np.array(original_parameters)

x1, x2, y = getData(eul, crp, das28)
crossValidations = []
for i in range(0, n_cv): crossValidations.append(crossValidation([x1, x2, y], i))

# Normalize each cross-validation set
for i, dataset in enumerate(crossValidations):
    [[x1t, x1v], [x2t, x2v], [yt, yv]] = dataset
    m = np.mean(x2t)
    std = np.std(x2t)
    x2t = (x2t - m) / std
    x2v = (x2v - m) / std
    crossValidations[i] = [[x1t, x1v], [x2t, x2v], [yt, yv]]

writeCsvHead(['learning_rate', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7', 'val_size'])

# This is run for as long as possible to test as many models as possible
while True:
    for i, parameter in enumerate(original_parameters):
        parameter_pool = augmentParameters(parameter, 1000)
        parameter_performance = {}
        # Go through all the augmented models
        for parameters in parameter_pool:
            performances = []
            # Go through all cross-validations
            for j in range(n_cv):
                [[x1t, x1v],[x2t, x2v],[yt, yv]] = crossValidation([x1, x2, y], j)
                performances.append(trainNetwork(parameters, x1t, x1v, x2t, x2v, yt, yv))
            new_loss = getPerformance(performances, window_size)
            parameter_performance[new_loss] = parameters
            if new_loss < min_loss:
                min_loss = new_loss
                print("New min loss:",new_loss,"Parameters:",parameters)
        best_keys = sorted(parameter_performance.keys())[:generation_size]
        new_parameters = [parameter_performance[x] for x in best_keys]
        original_parameters[i] = new_parameters
        # Save files in case of program interuption
        np.save('allgens/params_gen_{}_{}.npy'.format(generation, i), original_parameters)
        print("parameter done.")
    print("Generation", generation, "done.")
    generation += 1