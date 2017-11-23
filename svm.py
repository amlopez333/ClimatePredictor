from sklearn import svm
import numpy as np
from sklearn.model_selection import ShuffleSplit
import loadData as ld
import matplotlib.pyplot as plt
def computeError(prediction, actual):
    diff = 0
    for p, a in zip(prediction, actual):
        diff += 1 if p != a else 0
    return diff/len(actual) * 100

matrixOfData, labels = ld.getData('Datos Meteorologicos Manuales.xlsx', 6102)
print('loadedData')
splitter = ShuffleSplit(n_splits=1, test_size=0.1)
train = None
test = None
for tr, t in splitter.split(np.array(matrixOfData)):
    train = tr
    test = t

trainData = np.array([matrixOfData[i] for i in train])

trainLabels = [labels[i] for i in train]

print("Created training data of size: {}".format(len(trainData)))
testData = np.array([matrixOfData[i] for i in test])
testLabels = np.array([labels[i] for i in test])
clf = svm.SVC(kernel='rbf')
print('fitting data')
clf = clf.fit(trainData, trainLabels)
print('fitted data')
print("Percent correct: {}%".format(clf.score(testData, testLabels)*100))
print("Percent incorrect: {}%".format(computeError(clf.predict(testData), testLabels)))

'''# Put the result into a color plot
#Z = Z.reshape(XX.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(())
plt.yticks(())
fignum = 1 + 1

plt.show()'''
    