import keras
import numpy as np
from keras.layers import Input, concatenate, Dense
from keras.models import Model, Sequential
from keras.optimizers import Adam

# Constants
n_joints = 10
epochs = 300
batch_size = 128
n_cv = 5


def getData(eul, crp, das28):

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

def trainModel(parameters, x1t, x1v, x2t, x2v, yt, yv):

    x1t = keras.utils.to_categorical(x1t, 5)
    x1v = keras.utils.to_categorical(x1v, 5)

    x1t = np.array([i.flatten() for i in x1t])
    x1v = np.array([i.flatten() for i in x1v])


    # Hyperparameters
    learning_rate = parameters['learning_rate']
    dense_nodes = parameters['dense_nodes']

    # Reshape input
    x1t = np.reshape(x1t, (x1t.shape[0], x1t.shape[1]))
    x1v = np.reshape(x1v, (x1v.shape[0], x1v.shape[1]))

    # Input definition
    input_eular = Input(shape=(50,), dtype='float32', name='input_eular')
    input_crp = Input(shape=(1,), dtype='float32', name='input_crp')

    # Model definition
    x = concatenate([input_eular, input_crp], axis=1)
    for nodes in dense_nodes:
        x = Dense(nodes, activation='relu')(x)
    output = Dense(1)(x)

    model = Model(input=[input_eular, input_crp], output=output)

    optimizer = Adam(lr=learning_rate)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    cb_lrs = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=40, verbose=0)
    cbs = [cb_lrs]

    model.load_weights('best_weights.h5')

    hist = model.fit([x1t, x2t],
                     yt,
                     validation_data=([x1v, x2v], yv),
                     epochs=epochs,
                     batch_size=batch_size,
                     verbose=False,
                     callbacks=cbs)

    mae = hist.history['val_mean_absolute_error']
    mse = hist.history['val_loss']

    return mae, mse, model


# Data
eul = np.load("TrainingDataEUL.npy")
eul = np.array(eul, dtype='float32')
crp = np.load("TrainingDataCRP.npy")
das28 = np.load("TrainingDataY.npy")

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

params = {'learning_rate': 0.005, 'n_layers': 6, 'dense_nodes': [134, 189, 157, 220, 201, 235]}


for i in range(0, n_cv):
    [[x1t, x1v], [x2t, x2v], [yt, yv]] = crossValidations[i]
    mae, mse, model = trainModel(params, x1t, x1v, x2t, x2v, yt, yv)
    keras.models.save_model(model,'model_{}.h5'.format(i))
    print(mae[len(mae) - 1])
    print(mse[len(mse) - 1])