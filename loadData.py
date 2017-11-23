import numpy as np
from threading import Thread
from queue import Queue
from openpyxl import load_workbook

def formatData(rows, setOfCells, activeSheet, matrixOfData = []):
    for i in range(2, rows):
        rowOfData = [activeSheet['{0}{1}'.format(l, i)].value for l in setOfCells]
        for i in range(len(rowOfData)):
            try:
                rowOfData[i] = float(rowOfData[i])
            except ValueError:
                rowOfData[i] = 0

        matrixOfData.append(rowOfData)
    return matrixOfData
def labelData(matrixOfData, labels = []):
    for i in matrixOfData:
        rainfallLvl = i.pop()
        label = 1 if rainfallLvl > 0.0 else 0
        labels.append(label)
    return matrixOfData, labels

def getData(xlsWS, rows, setOfCells = ['F', 'G', 'I', 'K', 'Q', 'L'], allCells = False):
    wb = load_workbook(xlsWS)
    activeSheet = wb.active
    if not allCells:
        matrixOfData = formatData(rows, setOfCells, activeSheet)
        matrixOfData, labels = labelData(matrixOfData)
        return matrixOfData, labels