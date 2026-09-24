import matplotlib
import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-4, 4, 0.05)
print("Z Array:", z, len(z))


def sigmoid(z):
    f = 1 / (1 + np.exp(-z))
    d = f * (1 - f)
    return f, d



output, derivative = sigmoid(z)
plt.plot(z, output, label="sigmoid")
plt.plot(z, derivative, label="sigmoid derivative", color="red")
plt.xlabel("z")
plt.ylabel("f(z)")
plt.title("sigmoid")
plt.grid(True)
plt.legend()
plt.show()


def tanh(z):
    f=np.tanh(z)
    d = 1-f**2
    return f, d

output, derivative = tanh(z)
plt.plot(z, output, label="tanh")
plt.plot(z, derivative, label="tanh derivative", color="red")
plt.xlabel("z")
plt.ylabel("f(z)")
plt.title("tanh")
plt.grid(True)
plt.legend()
plt.show()


def relu(z):
    output=[]
    derivative=[]
    for i in z:
        if i>0:
            output.append(i)
            derivative.append(1)
        else:
            output.append(0)
            derivative.append(0)
    return output, derivative



output, derivative = relu(z)
plt.plot(z, output, label="relu")
plt.plot(z, derivative, label="relu derivative", color="red")
plt.xlabel("z")
plt.ylabel("f(z)")
plt.title("relu")
plt.grid(True)
plt.legend()
plt.show()


z=np.array([3.2,1.3,0.2,0.8])
def softmax(z):
    exponents=[np.exp(i) for i in z]
    s=sum(exponents)
    prob=[i/s for i in exponents]
    return prob
print("Prob=",softmax(z))