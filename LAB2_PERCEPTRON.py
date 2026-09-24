import numpy as np

X = np.array([[1,1],[1,0],[0,1],[0,0]])
T = np.array([-1,1,1,-1])
W = np.zeros(2, dtype=float)
b = 0
alpha = 1
theta = 15
epochs = 500

e = 1
while e <= epochs:
    print('Epoch=', e)
    y_pred = []
    for i in range(len(X)):
        x = X[i]
        t = T[i]
        Y_in = np.sum(x*W) + b
        if Y_in > theta:
            y_out = 1
        elif Y_in < -theta:
            y_out = -1
        else:
            y_out = 0
        y_pred.append(y_out)
        if y_out != t:
            b = b + alpha*t
            W = W + alpha*t*x
    print('w=', W, 'b=', b)
    y_pred = np.array(y_pred)
    if np.array_equal(T, y_pred):
        break
    e = e + 1

print('Found after'+ e + 'epochs')