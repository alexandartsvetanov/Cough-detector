# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sklearn
import sounddevice as sd
import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import random
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
def print_hi(name):

    plt.close('all')
    Fs = 160000
    d=3
    print("Start")
    a = sd.rec(int(d*Fs), Fs, 1, blocking=True)
    sd.wait()
    print("Stop")
    plt.plot(a); plt.title('recordrd sound')


    data = []


    res = []


    alist = list(a)
    alist.sort()
    #plt.plot(alist)
    #plt.show()
    proba = []
    rangeProb = 30000
    if a.argmax() <= 40000:
        rangeProb = a.argmax()
    if a.argmax() >= 470000:
        rangeProb = 510000 - a.argmax()
    maxIndex = a.argmax()
    for i in range(40000):
        proba.append(a[maxIndex - rangeProb + i])
    plt.plot(proba)
    data.append(max(proba) - min(proba))
    proba.sort()
    data.append(proba[10000])
    data.append(proba[30000])
    plt.plot(proba)
    #plt.show()


    #print(data)

    '''
    toZap = input("Da zapisvam li: ")
    isCaught = input("kashlqne li e: ")
    with open('data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if toZap == 'da':
            if isCaught == 'da':
                data.append(1)
            else:
                data.append(0)
            writer.writerow(data)
    '''
    trainx= []
    trainy = []
    testx = []
    testy = []
    num = 0
    dataread = pd.read_csv('data.csv')
    dataread = shuffle(dataread)
    trainx = dataread.iloc[0:15, [0, 1, 2]].values.reshape(-1, 3)
    trainy = dataread.iloc[0:15, 1].values.reshape(-1, 1)
    testx = dataread.iloc[15:25, [0, 1, 2]].values.reshape(-1, 3)
    testy = dataread.iloc[15:25, 3].values.reshape(-1, 1)
    return tryWithNeural(data)





model = Sequential()
model.add(Dense(12, input_dim=3, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
dataset = loadtxt('data.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:, 0:3]
y = dataset[:, 3]
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=150, batch_size=10)
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))
print("Davai")

def tryWithNeural(data):

    dataFormated = []
    for dat in data:
        dataFormated.append(float(dat))
    numArr = np.array(dataFormated)
    numArr = numArr.reshape((1, 3))
    #print(numArr)
    res = model.predict(numArr)
    #print(res)
    if res > 0.5:
        print("Covid!!!")
        return False
    else:
        print("Everything is fine")
        return True










if __name__ == '__main__':
    res  = True
    while res:
        res = print_hi('PyCharm')




