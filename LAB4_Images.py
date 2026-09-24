# import cv2
# import matplotlib.pyplot as plt

# image =  cv2.imread('tanjiro-kamado-red-3840x2160-22577.png')

# # cv2.imshow('Tanjiro', image)
# # cv2.waitKey(0)
# # plt.show()
# # print("Shape of Image",image.shape)
# # print("type of image",type(image))

# # plt.figure
# # plt.imshow(image)
# # plt.show()
# image1=cv2.resize(image,(224,224))
# image2=cv2.cvtColor(image1,cv2.COLOR_RGB2BGR)

# image3=cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
# plt.subplot(2,2,1)
# plt.imshow(image)
# plt.title("Original Image")

# plt.subplot(2,2,2)
# plt.imshow(image1)
# plt.title("resized Image")

# plt.subplot(2,2,3)
# plt.imshow(image2)
# plt.title("RGB2BGR")

# plt.subplot(2,2,4)
# plt.imshow(image3,cmap='gray')
# plt.title("Grayscale")

# plt.show()
import cv2
import csv
import os
labels={'daisy':0,'rose':1,'sunflower':2}
l=list(labels.keys())

with open("My flowers.csv",'w',newline='') as fl:
    for i in l:
        wr=csv.writer(fl)
        path='D:\\DATASET IMAGES\train\\'+i
        for r,d,f in os.walk(path):
            for fname in f:
                if '.jpg' or '.png' in fname:
                    record=[]
                    file_path=os.path.join(path,fname)
                    image1=cv2.imread(file_path)
                    image1=cv2.resize(image1,(300,200))
                    image1=image1.flatten()
                    label1=labels[i]
                    record.append(label1)   
                    record.extend(image1)
                    wr.writerow(record)