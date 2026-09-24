from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#one problem we need to fix next time is that we
#have to make sure that the csv contains the images stretched from multiple sides
#at different crops 

dataset = pd.read_csv(r'D:\5th Semester\Artificial Neural Networks and Deep Learning\Labs\dataset\28x28_rgb.csv')
X=dataset.drop(dataset.columns[0],axis=1)
Y=dataset[dataset.columns[0]]
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,random_state = 42,test_size = 0.2)

scaler=StandardScaler()
xtrain=scaler.fit_transform(xtrain)
xtest=scaler.transform(xtest)

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

ytrain = to_categorical(ytrain)
ytest  = to_categorical(ytest, num_classes=ytrain.shape[1])
num_classes = ytrain.shape[1]

model = Sequential([
    Dense(256, input_shape=(2352,)),
    Activation('relu'),
    Dense(512),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(128),
    Activation('relu'),
    Dense(3),
    Activation('softmax')

])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
summary=model.fit(xtrain,ytrain,epochs=15,batch_size=64,validation_split=0.2)
result=model.evaluate(xtest,ytest)
print ("loss=",result[0])
print ("accuracy=",result[1])
pd.DataFrame(summary.history).plot(figsize=(8,6))
plt.grid(True)
plt.show()

model.save('project_ann.keras')



# import os
# import cv2
# import numpy as np
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# from keras.models import load_model
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# BASE = r'D:\5th Semester\Artificial Neural Networks and Deep Learning\Labs'
# MODEL_PATH  = os.path.join(BASE, 'project_ann.keras')
# SCALER_PATH = os.path.join(BASE, 'scaler.pkl')
# CSV_PATH    = os.path.join(BASE, 'dataset', '28x28_rgb.csv')
# TEST_IMG    = os.path.join(BASE, 'test image for gan.png')
#
# CLASS_NAMES = {0: 'Diffusion', 1: 'GAN', 2: 'Faceswap'}
#
# model = load_model(MODEL_PATH)
#
# if os.path.exists(SCALER_PATH):
#     scaler = joblib.load(SCALER_PATH)
# else:
#     print("scaler.pkl not found, rebuilding from CSV (one time)...")
#     df = pd.read_csv(CSV_PATH)
#     X = df.drop(df.columns[0], axis=1)
#     Y = df[df.columns[0]]
#     xtrain, _, _, _ = train_test_split(X, Y, random_state=42, test_size=0.2)  # no stratify, matches training
#     scaler = StandardScaler().fit(xtrain)
#     joblib.dump(scaler, SCALER_PATH)
#     print("saved", SCALER_PATH)
#
#
# def _predict_bgr28(bgr28):
#     arr = bgr28.flatten().astype('float32').reshape(1, -1)
#     return model.predict(scaler.transform(arr), verbose=0)[0]
#
#
# def predict_image(path, show=True):
#     bgr = cv2.imread(path)
#     if bgr is None:
#         raise FileNotFoundError(path)
#     bgr = cv2.resize(bgr, (28, 28))
#     probs = _predict_bgr28(bgr)
#     top = int(probs.argmax())
#
#     print(f"\nPrediction: {CLASS_NAMES[top]}  ({probs[top]*100:.1f}% confidence)")
#     for i, p in enumerate(probs):
#         print(f"  {CLASS_NAMES[i]:<10} {p*100:6.2f}%")
#
#     if show:
#         fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
#         ax[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); ax[0].axis('off')
#         ax[0].set_title(f"{CLASS_NAMES[top]} ({probs[top]*100:.0f}%)")
#         ax[1].bar([CLASS_NAMES[i] for i in range(len(CLASS_NAMES))], probs * 100)
#         ax[1].set_ylabel('probability %'); ax[1].set_ylim(0, 100)
#         plt.tight_layout(); plt.show()
#     return top, probs
#
#
# def test_variants(path):
#     """Same image, three preprocessing styles. If predictions differ, the model is reacting to framing."""
#     img = cv2.imread(path)
#     if img is None:
#         raise FileNotFoundError(path)
#     h, w = img.shape[:2]
#
#     stretch = cv2.resize(img, (28, 28))
#
#     s = min(h, w)
#     y0, x0 = (h - s) // 2, (w - s) // 2
#     crop = cv2.resize(img[y0:y0+s, x0:x0+s], (28, 28))
#
#     s = max(h, w)
#     canvas = np.zeros((s, s, 3), dtype=img.dtype)
#     canvas[(s-h)//2:(s-h)//2+h, (s-w)//2:(s-w)//2+w] = img
#     pad = cv2.resize(canvas, (28, 28))
#
#     variants = {'stretch': stretch, 'center-crop': crop, 'black-pad': pad}
#     fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
#     for ax, (name, bgr) in zip(axes, variants.items()):
#         probs = _predict_bgr28(bgr)
#         top = int(probs.argmax())
#         print(f"{name:12s}", {CLASS_NAMES[i]: f"{p*100:.1f}%" for i, p in enumerate(probs)})
#         ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); ax.axis('off')
#         ax.set_title(f"{name}\n{CLASS_NAMES[top]} {probs[top]*100:.0f}%", fontsize=9)
#     plt.tight_layout(); plt.show()
#
#
# def test_on_dataset_rows(n=200):
#     """Pipeline sanity check: export real CSV rows as PNGs and confirm predict_image agrees with direct prediction."""
#     df = pd.read_csv(CSV_PATH).sample(n, random_state=0)
#     labels = df.iloc[:, 0].values
#     pixels = df.iloc[:, 1:].values.astype('float32')
#     direct = model.predict(scaler.transform(pixels), verbose=0).argmax(axis=1)
#     print(f"Accuracy on {n} random CSV rows (includes training rows): {(direct == labels).mean():.3f}")
#
#     # round-trip through PNG on 20 rows to confirm the image pipeline matches
#     agree = 0
#     for i in range(20):
#         img = pixels[i].astype('uint8').reshape(28, 28, 3)
#         tmp = os.path.join(BASE, '_tmp_row.png')
#         cv2.imwrite(tmp, img)
#         p = _predict_bgr28(cv2.resize(cv2.imread(tmp), (28, 28))).argmax()
#         agree += int(p == direct[i])
#     print(f"PNG round-trip agrees with direct prediction: {agree}/20")
#     os.remove(os.path.join(BASE, '_tmp_row.png'))

#
# if __name__ == '__main__':
#     predict_image(TEST_IMG)
#     test_variants(TEST_IMG)
#     # test_on_dataset_rows()     # uncomment for the pipeline sanity check