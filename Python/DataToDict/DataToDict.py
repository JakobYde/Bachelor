import csv
from numpy import asarray, save

joint_list = ['MCP1', 'MCP2', 'MCP3', 'MCP4', 'MCP5', 'PIP1', 'PIP2', 'PIP3', 'PIP4', 'PIP5']

def csv_to_dict(filename):
    d = {}
    with open(filename, mode='r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        rownames = []

        for i, row in enumerate(reader):
            if i == 0:
                for j, cell in enumerate(row):
                    rownames.append(row[j])
                    d[row[j]] = []
            else:
                for j, cell in enumerate(row):
                    d[rownames[j]].append(cell)
    return d

# Patient
# 'hands' - [[joints of hand 1],[joints of hand 2]]
# 'crp' - crp score
# 'das' - das score

def dict_to_patient(d):
    patients = []
    
    d['patient_number'] = [int(a) for a in d['patient_number']]
    d['hand_number'] = [int(a) for a in d['hand_number']]
    d['score'] = [float(a) for a in d['score']]

    n_patients = max([int(a) for a in d['patient_number']])
    j = 0

    for i in range(1, n_patients + 1):
        patient = {}
        patient['hands'] = [[],[]]
        for n in range(0, 10):
            patient['hands'][0].append(-1)
            patient['hands'][1].append(-1)
        patient['crp'] = -1
        patient['das'] = -1

        while j < len(d['patient_number']) and d['patient_number'][j] == i:
            if d['joint_name'][j] == 'CRP':
                patient['crp'] = d['score'][j]
            elif d['joint_name'][j] == 'DAS':
                patient['das'] = d['score'][j]
            else:               
                patient['hands'][d['hand_number'][j] - 1][joint_list.index(d['joint_name'][j])] = d['score'][j]
            j += 1
        patients.append(patient)
    return patients

def patients_to_nparrays(patients):
    das28 = []
    crp = []
    eul = []
    for patient in patients:
        eul.append(patient['hands'][0])
        eul.append(patient['hands'][1])
        das28.append(patient['das'])
        crp.append(patient['crp'])
        crp.append(patient['crp'])
    eul = asarray(eul)
    crp = asarray(crp)
    das28 = asarray(das28)
    return eul, crp, das28

d = csv_to_dict('NewNewData.csv')
patients = dict_to_patient(d)
eul, crp, das = patients_to_nparrays(patients)

save('NewData_eul.npy',eul)
save('NewData_crp.npy',crp)
save('NewData_das.npy',das)
