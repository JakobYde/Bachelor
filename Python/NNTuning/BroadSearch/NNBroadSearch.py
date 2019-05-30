import numpy as np
import keras as ke
import csv

from keras import models, layers, optimizers

def normalize(x_train, x_val):
    mu = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    x_train_normalized = (x_train - mu) / std
    x_val_normalized = (x_val - mu) / std

    return x_train_normalized, x_val_normalized

def remove_duplicates(array):
    array_no_duplicates = []
    for element in array:
        if element not in array_no_duplicates:
            array_no_duplicates.append(element)
    return array_no_duplicates

# Load data
xTr = np.load("TrainingDataX.npy")
yTr = np.load("TrainingDataY.npy")
NNmodels = np.load("Models.npy")

# Split data for cross-validation
x = np.array(np.array_split(xTr, 5))
y = np.array(np.array_split(yTr, 5))

z = 0
x_normalized = []
for i, x_va in enumerate(x):

# Normalize every cross-validation set
    x_train = np.concatenate(np.copy(x)[np.arange(len(x)) != i])
    y_train = np.concatenate(np.copy(y)[np.arange(len(y)) != i])

    x_val = np.copy(x_va)

    CRP_train = [x_train[j][50] for j in range(len(x_train))]
    CRP_val = [x_val[j][50] for j in range(len(x_val))]

    CRP_train, CRP_val = normalize(CRP_train, CRP_val)

    for j in range(len(x_train)):
        x_train[j][50] = CRP_train[j]

    for j in range(len(x_val)):
        x_val[j][50] = CRP_val[j]

    x_normalized.append([x_train, x_val])

x_normalized = np.array(x_normalized)


for NN in NNmodels:
    for i, x_norm in enumerate(x_normalized):

        y_train = np.concatenate(y[np.arange(len(y)) != i])
        y_val = y[i]

        model = models.Sequential()

        model.add(layers.Dense(NN['dense_nodes'][0], activation='relu', input_shape=(51,)))

        s = '{},{},'.format(NN['learning_rate'], NN['dense_nodes'][0])

        if len(NN) > 1:
            for layer in NN['dense_nodes'][1:]:
                model.add(layers.Dense(layer, activation='relu'))
                s += '{},'.format(layer)

        for j in range(7-len(NN['dense_nodes'])):
            s += '0,'

        s += '{},:,val_mae,'.format(len(x_norm[1]))

        model.add(layers.Dense(1))

        print(s)

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        hist = model.fit(x_norm[0], y_train, batch_size=128, epochs=100, validation_data=(x_norm[1], y_val), verbose=False)
        ke.backend.clear_session()

        mae_arr = hist.history['val_mean_absolute_error']
        mse_arr = hist.history['val_loss']
        mse_train = hist.history['loss']

        print("done. results: {}_{}".format(str(min(mae_arr)), min(mse_arr)), i, z)
        z += 1

        with open('NNresults.csv', mode='a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow([s, [round(float(n), 2) for n in mae_arr], ':,val_mse,', [round(float(n), 2) for n in mse_arr], ':,train_mse,', [round(float(n), 2) for n in mse_train]])
pass