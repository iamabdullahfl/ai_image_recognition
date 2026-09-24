from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

dataset=load_breast_cancer()
X=dataset.data
y=dataset.target
y=np.where(y==0,-1,y)
W=np.zeros(30,dtype=float)
#W=np.array([0,0],dtype=float)
b=0
alpha=1
theta=0.1
epochs=500
e=1
Y_in=y
while e<=epochs:
    print('Epoch=',e)
    y_pred=[]
    for i in range (len(X)):
      x=X[i]
      t=y[i]
      Y_in=np.sum(x*W)+b
      if Y_in>theta:
          Y=1
      elif Y_in < -theta:
          Y=-1
      else:
          Y=0
      y_pred.append(Y)
      if Y!=t:
          b=b+alpha*t
          W=W+alpha*t*x
    print ('w=',W,'b',b)
    y_pred=np.array(y_pred)
    if np.array_equal(y,y_pred):
        break
    e=e+1
print('Accuracy:',accuracy_score(y,y_pred))




X=np.array[[1,1],[1,0],[0,1],[0,0]]
Y=np.array([1,1,1,-1])
W=np.zeros(2,dtype=float)
#W=np.array([0,0],dtype=float)
b=0
alpha=1
theta=15
epochs=500
e=1
while e<=epochs:
    print('Epoch=',e)
    y_pred=[]
    for i in range (len(X)):
      x=X[i]
      t=Y[i]
      Y_in=np.sum(x*W)+b
      if Y_in>theta:
          Y=1
      elif Y_in < -theta:
          Y=-1
      else:
          Y=0
      y_pred.append(Y)
      if Y!=t:
          b=b+alpha*t
          W=W+alpha*t*x
    print ('w=',W,'b',b)
    y_pred=np.array(y_pred)
    if np.array_equal(Y,y_pred):
        break
    e=e+1
