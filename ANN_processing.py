# from keras.models import Sequential
# from keras.layers import Dense,Activation
# from keras.utils import to_categorical
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# #one problem we need to fix next time is that we
# #have to make sure that the csv contains the images stretched from multiple sides
# #at different crops
#
# dataset = pd.read_csv(r'D:\5th Semester\Artificial Neural Networks and Deep Learning\Labs\dataset\28x28_rgb.csv')
# X=dataset.drop(dataset.columns[0],axis=1)
# Y=dataset[dataset.columns[0]]
# xtrain,xtest,ytrain,ytest = train_test_split(X,Y,random_state = 42,test_size = 0.2)
#
# scaler=StandardScaler()
# xtrain=scaler.fit_transform(xtrain)
# xtest=scaler.transform(xtest)
#
# print(xtrain.shape)
# print(xtest.shape)
# print(ytrain.shape)
# print(ytest.shape)
#
# ytrain = to_categorical(ytrain)
# ytest  = to_categorical(ytest, num_classes=ytrain.shape[1])
# num_classes = ytrain.shape[1]
#
# model = Sequential([
#     Dense(1024, input_shape=(2352,)),
#     Activation('relu'),
#     Dense(512),
#     Activation('relu'),
#     Dense(256),
#     Activation('relu'),
#     Dense(128),
#     Activation('relu'),
#     Dense(3),
#     Activation('softmax')
#
# ])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# summary=model.fit(xtrain,ytrain,epochs=15,batch_size=64,validation_split=0.2)
# result=model.evaluate(xtest,ytest)
# print ("loss=",result[0])
# print ("accuracy=",result[1])
# pd.DataFrame(summary.history).plot(figsize=(8,6))
# plt.grid(True)
# plt.show()
#
# model.save('project_ann.keras')


dataset= pd.read_csv(path)
pixels=dataset.iloc[:0]

img=pixels.astype("float32")
