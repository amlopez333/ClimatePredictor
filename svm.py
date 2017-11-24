from sklearn import svm
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
import loadData as ld
from threading import Thread
from queue import Queue
import matplotlib.pyplot as plt
def computeError(prediction, actual):
    diff = 0
    for p, a in zip(prediction, actual):
        diff += 1 if p != a else 0
    return diff/len(actual) * 100

matrixOfData, labels, measures = ld.processAll()
print('loadedData')
splitter = ShuffleSplit(n_splits=1, test_size=0.1)
train = None
test = None
for tr, t in splitter.split(np.array(matrixOfData)):
    train = tr
    test = t

trainData = np.array([matrixOfData[i] for i in train])

trainLabels =np.array([labels[i] for i in train])

trainMeasures = np.array([measures[i] for i in train])
scaler = preprocessing.StandardScaler().fit(trainData)
normTrainData = scaler.transform(trainData)
print("Created training data of size: {}".format(len(trainData)))
testData = np.array([matrixOfData[i] for i in test])
normTestData = scaler.transform(testData)
testLabels = np.array([labels[i] for i in test])
testMeasures = np.array([measures[i] for i in test])

print(len(normTrainData[0]))

class SVCWorker(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue
        self.daemon = True

    def run(self):
        while True:
            svc, kernel, trainData, trainMeasures = self.queue.get()
            print('fitting data for {}'.format(kernel))
            svc = svc.fit(normTrainData, trainLabels)
            print('fitted data for {}'.format(kernel))
            print("Percent correct using {1} kernel: {0}%".format(svc.score(normTestData, testLabels)*100, kernel))
            print("Percent incorrect using {1} kernel: {0}%".format(computeError(svc.predict(normTestData), testLabels), kernel))
            self.queue.task_done()
    

print('Creating rbf kernel svm')
rbfCLF = svm.SVC(kernel='rbf')

print('Creating poly with degree = 3 kernel svm')
polyCLF = svm.SVC(kernel='poly')

print('Creating linear kernel svm')
linearCLF = svm.SVC(kernel='linear')
queue = Queue()
for x in range(3):
    svcw = SVCWorker(queue)
    svcw.start()
queue.put((rbfCLF, 'rbf', normTrainData, trainMeasures))
#queue.put((linearCLF, 'linear', normTrainData, trainMeasures))
#queue.put((polyCLF, 'poly', normTrainData, trainMeasures))
queue.join()

class SVRWorker(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue
        self.daemon = True

    def run(self):
        while True:
            svr, normTrainData, trainMeasures = self.queue.get()
            svr = svr.fit(normTrainData, trainMeasures).predict(trainData)
            print(svr, trainMeasures)
            input('Press to continue')
            self.queue.task_done()
    

svr_rbf = svm.SVR(kernel='rbf')
svr_lin = svm.SVR(kernel='linear')
svr_poly = svm.SVR(kernel='poly')
queue = Queue()
for x in range(3):
    svrw = SVRWorker(queue)
    svrw.start()

queue.put((svr_rbf, normTrainData, trainMeasures))
#queue.put((svr_lin, normTrainData, trainMeasures))
#queue.put((svr_poly, normTrainData, trainMeasures))

queue.join() 