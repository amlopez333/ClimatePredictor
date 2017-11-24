import numpy as np
from threading import Thread
from queue import Queue
import csv

class Worker(Thread):
    def __init__(self, inQueue, outQueue):
        Thread.__init__(self)
        self.inQueue = inQueue
        self.outQueue = outQueue
        self.daemon = True
        self.fieldSet = ['AIRTEMP_C_AVG', 'WINDSP_MS_U', 'WINDDIR_DU', 'BP_MBAR', 'RAIN_MM_TOT']
    def run(self):
        while True:
            row = self.inQueue.get()
            dataRow = [row[i] for i in self.fieldSet]
            self.outQueue.put(dataRow)
            self.inQueue.task_done()

def getData():
    inQueue = Queue()
    outQueue = Queue()

    for x in range(8):
        worker = Worker(inQueue, outQueue)
        worker.start()
    with open('Datos Meteorologicos Automaticos.csv') as dataCSV:
        reader = csv.DictReader(dataCSV)
        for row in reader:
            inQueue.put(row)
        inQueue.join()
    validData = []

    while not (outQueue.empty()):
        row = outQueue.get()
        valid = True
        for i in row:
            if i == '':
                valid = False
                break
        if(valid):
            validData.append(row)
    return validData
def processValidData(validData, processedData = []):
    for row in validData:
        newRow = [float(j) for j in row]
        processedData.append(newRow)
    return processedData

def getProcessedData():
    validData = getData()
    processedData = processValidData(validData)
    return processedData

def labelData(processedData, labels = []):
    for elem in processedData:
        rainfallLvl = elem.pop()
        if (rainfallLvl > 0.0):
            labels.append(1)
        else:
            labels.append(-1)
    return labels

def getMeasures(processedData, measures = []):
    for elem in processedData:
        rainfallLvl = elem.pop()
        measures.append(rainfallLvl)
    return measures

def createFeatureMatrix(processedData, featureMatrix = []):
    for elem in processedData:
        rainfallLvl = elem.pop()
        featureMatrix.append(elem)
    return featureMatrix

def processAll():
    processedData = getProcessedData()
    labels = labelData(processedData)
    measures = getMeasures(processedData)
    featureMatrix = createFeatureMatrix(processedData)
    return featureMatrix, labels, measures