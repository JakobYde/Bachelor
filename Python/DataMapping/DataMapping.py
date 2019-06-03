import numpy as np
import csv
from Session import Session
from ImageInformation import ImageInformation
from keras.utils import to_categorical

dateDict = dict()

filename = 1
contentdate = 2
contenttime = 3
seriesdate = 4
seriesnumber = 5
seriestime = 6
studydate = 7
studytime = 8
joint_scanned = 9
score = 10

UL_PD_scores = [] 
DAS28scores = []

def normalize(x, mean="default", std="default"):
    if mean is "default": mean = np.average(x)
    if std is "default": std = np.std(x)
    return (x - mean) / std

with open('UL_PD_scores.csv', 'rt') as csvfile:
    file = csv.reader(csvfile, delimiter=',')
    UL_PD_scores = list(file)

with open('DAS28data.csv', 'rt') as csvfile:
    file = csv.reader(csvfile, delimiter=',')
    DAS28scores = list(file)

for row in UL_PD_scores[1:]:
    file = row[filename]
    name = file[3:5]
    date = row[contentdate]
    joint = row[joint_scanned]

    if joint in Session.jointNames:
        if dateDict.get(date) is None:
            sess = Session()
            sess.name = name
            sess.add_file(file, joint, int(row[score]))
            dateDict[date] = [sess]
            sess.calc_avg()
        else:
            if dateDict[date][len(dateDict[date]) - 1].name != name:
                dateDict[date].append(Session())
            currentSess = dateDict[date][len(dateDict[date]) - 1]
            currentSess.name = name
            currentSess.add_file(file, joint, int(row[score]))
            currentSess.calc_avg()


DAS28scores.sort(key=lambda x: (float(x[0]), (float(x[2]) - 0.36 * np.log(float(x[1]) + 1))))

last_date = ""
DAS28_date_dict = dict()
patients_on_date = 1
for row in DAS28scores:
    date = row[0]
    if date == last_date:
        last_date = date
        patients_on_date += 1
    else:
        last_date = date
        patients_on_date = 1
    DAS28_date_dict[date] = patients_on_date

i = 0

while i < DAS28scores.__len__():
    date = DAS28scores[i][0]

    lookup = dateDict.get(date)
    if lookup is not None:
        if lookup.__len__() == DAS28_date_dict.get(date):
            if lookup.__len__() > 1:
                lookup.sort(key=lambda x: x.avg)

            for sess in lookup:
                sess.CRP = DAS28scores[i][1]
                sess.DAS28 = DAS28scores[i][2]

                i += 1
        else:
            i += 1
            print("UL_PD: ", lookup.__len__(), " DAS28: ", DAS28_date_dict.get(date), " Date: ", date)
            dateDict[date] = []
    else:
        i += 1

with open('sessions.csv', mode='w+') as file:
    writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(['Duplicates', 'Missing', 'Perfect'])

perfect = 0
missing_data = 0
duplicates = 0

joints = 0

for key, date in dateDict.items():
    for sess in date:
        (status, n_joints) = sess.print_to_file('sessions.csv')
        joints += n_joints
        if status[2]:
            perfect += 1
        else: 
            if status[1]:
                missing_data += 1

            elif status[0]:
                duplicates += 1

with open('sessions.csv', mode='a') as file:
    writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow([duplicates, missing_data, perfect, joints])

x1 = []
x2 = []
y = []

for key, date in dateDict.items():
    for sess in date:
        imgs = []
        for image in sess.files:
            imgs.append(int(image.score))
        x1.append(imgs)
        x2.append(int(sess.CRP))
        y.append(float(sess.DAS28) / 10.0)
   
x2 = np.asarray(x2)
x2.shape = (116)

x = to_categorical(x1, 5)
x = np.array([a.flatten() for a in x])
x = np.hstack((x,np.reshape(x2,(116,1))))

print(x)
y = np.array(y)

indeces = np.array(range(116))
np.random.shuffle(indeces)

y_ = y
y = np.array([y_[i] for i in indeces])

x_ = x
x = np.array([x_[i] for i in indeces])

x1_ = x1
x1 = np.array([x1_[i] for i in indeces])

x2_ = x2
x2 = np.array([x2_[i] for i in indeces])

nTr = int(len(y) * 0.85)
nTe = len(y) - nTr

cTr = nTr

xTr = x[:cTr]
xTe = x[nTr:]

yTr = y[:cTr]
yTe = y[nTr:]

eulTr = x1[:cTr]
eulTe = x1[nTr:]

crpTr = x2[:cTr]
crpTe = x2[nTr:]

crpMeans = [np.average(x) for x in [crpTr, crpTe]]
crpSigmas = [np.std(x) for x in [crpTr, crpTe]]

np.save("TrainingDataX.npy",xTr)
np.save("TestingDataX.npy",xTe)

np.save("TrainingDataEUL.npy",eulTr)
np.save("TestingDataEUL.npy",eulTe)

np.save("TrainingDataCRP.npy",crpTr)
np.save("TestingDataCRP.npy",crpTe)

np.save("TrainingDataY.npy",yTr)
np.save("TestingDataY.npy",yTe)

np.save("Indeces.npy",indeces)

pass