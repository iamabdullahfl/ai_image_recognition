# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.utils import to_categorical
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Activation
# import matplotlib.pyplot as plt
# import pandas as pd
#
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)
#
# print(y_train)
#
# plt.figure()
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     plt.imshow(X_train[i],cmap='gray')
# plt.show()
#
# X_train = X_train.reshape(60000,784)
# X_test = X_test.reshape(10000,784)
# y_train = to_categorical(y_train,10)
# testY=to_categorical(y_test,10)
#
# model = Sequential(
#     [
#         Dense(999,input_dim=784),
#         Activation('relu'),
#         Dense(999,input_dim=999),
#         Activation('relu'),
#         Dense(10),
#         Activation('softmax')
#     ]
# )
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# summary=model.fit(X_train,y_train,epochs=10,batch_size=128)
# result=model.evaluate(X_test,y_test)
# print ("loss=",result[0])
# print ("accuracy=",result[1])
# pd.DataFrame(summary.history).plot(figsize=(8,6))
# plt.grid(True)
# plt.show()
# model.save('mnist_trained.keras')
#
#
#
#
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Activation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

print(y_train)

plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i],cmap='gray')
plt.show()

X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)
y_train = to_categorical(y_train,10)
testY=to_categorical(y_test,10)

model = Sequential(
    [
        Dense(30,input_dim=784),
        Activation('relu'),

        Dense(10),
        Activation('softmax')
    ]
)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
summary=model.fit(X_train,y_train,epochs=10,batch_size=128)
result=model.evaluate(X_test,testY)
print ("loss=",result[0])
print ("accuracy=",result[1])
pd.DataFrame(summary.history).plot(figsize=(8,6))
plt.grid(True)
plt.show()
model.save('mnist_trained.keras')

img = Image.open(r'mnist_image2.png').convert('L')
img = img.resize((28, 28))
arr = np.array(img).astype('float32')

if arr.mean() > 127:
    arr = 255 - arr

plt.imshow(arr, cmap='gray')
plt.title('What the model sees')
plt.axis('off')
plt.show()

x = arr.reshape(1, 784)
probs = model.predict(x)[0]
print("Prediction:", np.argmax(probs))
print("Confidence:", round(float(np.max(probs)) * 100, 2), "%")
model.save('mnist_trained.keras')