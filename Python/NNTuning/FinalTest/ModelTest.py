import numpy as np
import keras
from keras.models import load_model
import os

eul = np.load("TestingDataEUL.npy")
eul = np.array(eul, dtype='float32')
eul = np.reshape(eul, (eul.shape[0], 10, 1))
crp = np.load("TestingDataCRP.npy")
crp = np.reshape(crp, (crp.shape[0], 1))
train_crp = np.load("TrainingDataCRP.npy")
train_crp = np.reshape(train_crp, (train_crp.shape[0], 1))
das28 = np.load("TestingDataY.npy")

eul = keras.utils.to_categorical(eul, 5)
eul = np.array([i.flatten() for i in eul])
eul = np.reshape(eul, (eul.shape[0], eul.shape[1]))

crp = (crp - np.mean(train_crp)) / np.std(train_crp)

ensemble_dir = r'C:\Users\Jakobs PC\Desktop\Bachelor\NNFinalTest\models/'

model_dir = r'C:\Users\Jakobs PC\Desktop\Bachelor\NNFinalTest\model/'

def loadModels(eul=eul, crp=crp, das28=das28, dir=ensemble_dir):
    models = []
    all_preds = []

    for i, file in enumerate(os.listdir(dir)):
        print('Started model {}. ({})'.format(i, file))
        model = load_model(dir + file)
        preds = model.predict([eul, crp])
        model = None
        keras.backend.clear_session()
        all_preds.append(preds)
        diff = preds - das28

        m = {}
        m['name'] = file
        m['mae'] = np.mean(np.abs(diff))
        m['mse'] = np.mean(np.square(diff))
        models.append(m)
        print('Model',i,'done.',m)

    all_preds = np.mean(all_preds, 0)
    all_preds = np.resize(all_preds, len(all_preds))
    print(all_preds)
    diff = all_preds - das28
    mae = np.mean(np.abs(diff))
    mse = np.mean(np.square(diff))
    return mae, mse, models

mae, mse, models = loadModels()
print(mae, mse)
pass