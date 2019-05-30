import numpy as np
import csv

x_training_eul = np.load("TrainingDataEUL.npy")
x_test_eul = np.load("TestingDataEUL.npy")
x_training_crp = np.load("TrainingDataCRP.npy")
x_test_crp = np.load("TestingDataCRP.npy")
y_training = np.load("TrainingDataY.npy")
y_test = np.load("TestingDataY.npy")

np.savetxt("TrainingDataEUL.csv", x_training_eul, delimiter=",")
np.savetxt("TestingDataEUL.csv", x_test_eul, delimiter=",")
np.savetxt("TrainingDataCRP.csv", x_training_crp, delimiter=",")
np.savetxt("TestingDataCRP.csv", x_test_crp, delimiter=",")
np.savetxt("TrainingDataY.csv", y_training, delimiter=",")
np.savetxt("TestingDataY.csv", y_test, delimiter=",")
