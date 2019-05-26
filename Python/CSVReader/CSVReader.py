import csv
import numpy as np

cfn = 5 # Cross-fold n

csv.register_dialect('myDialect',
                     delimiter=';')

def filterArray(array, n):
    result = []
    d = int(n / 2)
    for i in range (d, len(array) - d):
        result.append(np.mean(array[i - d: i + d + 1]))
    return result

def saveFile(to_print, filename, delimiter):
    with open(filename, mode='w',newline='') as employee_file:
        writer = csv.writer(employee_file, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for row in to_print:
            writer.writerow(row)

def filterFile(filename):
    result = []
    head = []
    data = []
    weights = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)

        categories = []
        
        for i, row in enumerate(reader):
            if i is 0:
                for cell in row: categories.append(cell)
            else:
                i -= 1
                r = i / cfn # Result row
                id = i % cfn # Data indeks 
        
                if id is 0:
                    weightsum = 0
                    data = []
                    weights = []

                rdata = {}
                
                c_head = row.index(':')
                head = row[:c_head]
                row = row[c_head + 1:]

                while ':' in row:
                    index = row.index(':')
                    d = row[:index]
                    row = row[index + 1:]
                    rdata[d[0]] = d[1:]

                rdata[row[0]] = row[1:]
                data.append(rdata)

                weight = int(head[-1])
                weightsum += weight
                weights.append(weight)

                if id is cfn - 1:
            
                    min_vals = {}

                    m = len(data)
                    for key in data[0]:
                        n = len(data[0][key])
                        filter_array = [0.] * n
                        
                        for nr, r in enumerate(data):
                            for i, v in enumerate(r[key]):
                                filter_array[i] += float(v) * weights[nr]

                        for i in range(0, n):
                            filter_array[i] /= weightsum

                        filtered = filterArray(filter_array, 5)
                        min_vals[key] = np.min(filtered)

                    head[-1] = weightsum

                    for key in min_vals:
                        if ('min_' + key) not in categories: categories.append('min_' + key)
                        head.append('{:.4f}'.format(min_vals[key]))

                    result.append(head)

    return categories, result


categories, result = filterFile('resultsfinetuneNN.csv')
to_print = [categories] + result
saveFile(to_print, "filtered.csv", ",")