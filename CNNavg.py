import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from keras import models
from keras.layers import Input, Conv2D, BatchNormalization, Activation, SpatialDropout2D, AveragePooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("Images28.csv")

x = data.iloc[:, 1:].to_numpy(dtype="float32").reshape(-1, 28, 28, 3)
y = data.iloc[:, 0].to_numpy(dtype="int32")

print("Dataset shape:", x.shape)
print("Labels:", np.unique(y))
print("Class counts:", np.bincount(y))


Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

Xtrain = Xtrain / 255.0
Xtest = Xtest / 255.0

print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)


model = models.Sequential([
    Input(shape=(28, 28, 3)),

    Conv2D(16, (3, 3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(16, (3, 3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    AveragePooling2D((2, 2)),
    SpatialDropout2D(0.05),

    Conv2D(32, (3, 3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(32, (3, 3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    AveragePooling2D((2, 2)),
    SpatialDropout2D(0.08),

    Conv2D(64, (3, 3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(64, (3, 3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    AveragePooling2D((2, 2)),
    SpatialDropout2D(0.10),

    Conv2D(128, (3, 3), padding="same"),
    BatchNormalization(),
    Activation("relu"),

    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.40),
    Dense(3, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()


history = model.fit(Xtrain, ytrain, batch_size=32, epochs=100, validation_split=0.15, verbose=1)

result = model.evaluate(Xtest, ytest)
print("loss=", result[0])
print("accuracy=", result[1])
pd.DataFrame(history.history).plot(figsize=(8, 6))
plt.grid(True)
plt.show()



model.save("CNNavg.keras")

plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()


model = tf.keras.models.load_model("CNNavg.keras")


loss, accuracy = model.evaluate(Xtest, ytest, verbose=0)

print("\n\nFINAL TEST RESULTS")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")


predictions = model.predict(Xtest, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)


print("\nClassification Report:")
print(classification_report(ytest, predicted_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(ytest, predicted_labels))