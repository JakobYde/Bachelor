import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as go

import numpy as np
import csv

csv.register_dialect('myDialect',
                     delimiter=';')

def loadData(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        categories = []

        for i, row in enumerate(reader):
            if i is 0:
                for cell in row: categories.append(cell)
            else:
                rowDict = {}
                for j, cell in enumerate(row): rowDict[categories[j]] = float(cell)
                data.append(rowDict)
    return data

def drawHistogram(data, accuracyName, variableName):
    x = [d[variableName] for d in data]
    y = [d[accuracyName] for d in data]
    colorscale = ['rgb(255,0,0)', (1,1,1)]

    f = ff.create_2d_density(
        x, y, colorscale=colorscale,
        hist_color='rgb(255, 237, 222)', point_size=3,
        title='',
        height=1080,
        width=1920)
    return f

# a
def drawParallelCoords(data, accuracyName, dimensionNames, lowerisbetter=True):
    accuracies = [d[accuracyName] for d in data]
    accuracy_dim = go.parcats.Dimension(
      values=accuracies,
      label="mae",
    )

    if lowerisbetter:
        cmin = np.min(accuracies)
        cmax = np.mean(accuracies) + np.std(accuracies)
    else:
        cmin = np.mean(accuracies) - np.std(accuracies)
        cmax = np.max(accuracies)


    cs =  [int(acc / np.max(accuracies) * 255.) for acc in accuracies]
    colors = ['rgb({},0,0)'.format(c) for c in cs]

    dimensionList = []
    for dimensionName in dimensionNames:
        d = [d[dimensionName] for d in data]
        dimensionList.append(dict(range=[np.min(d), np.max(d)], label=dimensionName,values=d))
    dimensionList.append(dict(range=[np.min(accuracies), np.max(accuracies)],label=accuracyName,values=accuracies))

    data = [go.Parcoords(
            line = dict(color = accuracies,
                       colorscale = 'Bluered',
                       reversescale=lowerisbetter,
                       showscale = True,
                       cmin = cmin,
                       cmax = cmax),
            dimensions = list(dimensionList))]
    return data
#b
def visualizeData(data, accuracyNames, filenamePrefix = '', excludeNames = [], maxAccuracy = [], lowerisbetter=True):
    keys = [key for key in data[0]]

    for d in data:
        for key in keys:
            if key not in d:
                keys.remove(key)

    for accuracy in accuracyNames: keys.remove(accuracy)
    for exclude in excludeNames: 
        if exclude in keys: 
            keys.remove(exclude)

    isCategory = [isinstance(data[0][key], str) for key in keys]
    categoryList = [None] * len(isCategory)
    for i, b in enumerate(isCategory):
        if b:
            l = list(set([d[keys[i]] for d in data]))
            for j in range(0, len(data)):
                data[j][keys[i]] = l.index(data[j][keys[i]])
            categoryList[i] = l
            pass

    for accuracyName in accuracyNames:
        for variableName in keys:
            plot = drawHistogram(data, accuracyName, variableName)
            filename = 'hist_{}_{}'.format(accuracyName, variableName)
            if filenamePrefix is not '': filename = '{}_{}'.format(filenamePrefix, filename)
            py.plot(plot,filename=filename,auto_open=False)

        
    for i, accuracyName in enumerate(accuracyNames):
        filename = 'parcor_{}'.format(accuracyName)
        if filenamePrefix is not '': filename = '{}_{}'.format(filenamePrefix, filename)
        p = drawParallelCoords(data, accuracyName, keys, lowerisbetter)
        py.plot(p, filename=filename,auto_open=False)

def getStats(data, to_measure = []):
    result = {}
    for m in to_measure:
        stats = {}
        d = [dat[m] for dat in data]
        stats['stdev'] = np.std(d)
        stats['mean'] = np.mean(d)
        result[m] = stats
    return result


files = ['inc_filtered.csv']
stats = []

for i, f in enumerate(files):
    data = loadData(f)
    visualizeData(data, ['min_val_mae', 'min_val_mse', 'min_train_mse'], f[:len(f)-4], ['val_size', 'learning_rate'], [], True)
    stats.append(getStats(data, ['min_val_mae', 'min_val_mse', 'min_train_mse']))

for s in stats: print(s)

