import cv2
from matplotlib import pyplot as plt
image1 = cv2.imread('image.jpg')
cv2.imshow('image', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Shape of image: ", image1.shape)
print("Type of image:", type(image1))

image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(image1_rgb)
plt.title('image1')
plt.show()

image2 = cv2.imread('img.jpg')
image2 = cv2.resize(image2, (500, 500))

image3 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)   # BGR -> RGB (correct colors)
image4 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)  # BGR -> Grayscale

plt.figure(figsize=(10, 10))

plt.subplot(3, 2, 1)
plt.imshow(image1_rgb)
plt.title('image1 (RGB)')

plt.subplot(3, 2, 2)
plt.imshow(image2)
plt.title('image2 (BGR - colors look off)')

plt.subplot(3, 2, 3)
plt.imshow(image3)
plt.title('image3 (RGB - correct colors)')

plt.subplot(3, 2, 4)
plt.imshow(image4, cmap='gray')
plt.title('image4 (grayscale)')

plt.subplot(3, 2, 5)
plt.imshow(image4, cmap='gray')
plt.title('image4 (grayscale, duplicate)')

plt.tight_layout()
plt.show()

#we need to display 4 images at once


import csv
import os
labels={
    'a':0,
    'b':1,
    'c':2,
}
l=list(labels.keys())#['a','b','c']
with open("myalphabets.csv",'w',newline='') as fl:
    for i in l:
        wr=csv.writer(fl)
        path='path dein gay yaha pr'+i
        for r,d,f in os.walk(path):
            for fname in f:
                if '.jpg' or '.png' or '.jpeg' in fname:
                    record=[]
                    file_path=os.path.join(path,fname)
                    image1=cv2.imread(file_path)
                    image1=cv2.resize(image1, (224, 224))
                    image1=image1.flatten()
                    label1=labels[i]
                    record.append(label1)
                    record.extend(image1)
                    wr.writerow(record)
