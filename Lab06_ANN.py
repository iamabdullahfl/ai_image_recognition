import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
data=pd.read_csv('heart.csv')
X=data.drop(['output'],axis=1)
Y=data['output']
print("X=",X.shape)
print("Y=",Y.shape )
sc=MinMaxScaler()
X1=sc.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X1,Y,test_size=0.2,random_state=42)
NNmodel=Sequential()
NNmodel.add(Dense(units=13,activation='sigmoid',input_shape=(13,)))
NNmodel.add(Dense(991,activation='relu'))
NNmodel.add(Dense(1,activation='sigmoid'))
NNmodel.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
summary=NNmodel.fit(X_train,y_train,epochs=5000,batch_size=32)
result=NNmodel.evaluate(X_test,y_test)
predictions=NNmodel.predict(X_test)
print ("loss=",result[0])
print ("accuracy=",result[1])
pd.DataFrame(summary.history).plot(figsize=(8,6))
plt.grid(True)
plt.show()





