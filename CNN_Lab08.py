from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape) #(60000,28,28)
print(y_train.shape) #(10000,28,28)
Ytrain=to_categorical(y_train,10)
Ytest=to_categorical(y_test,10)
Xtrain=x_train.reshape((60000,28,28,1))
Xtest=x_test.reshape((10000,28,28,1))
