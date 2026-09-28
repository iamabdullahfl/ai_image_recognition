import matplotlib as plt
import pandas as pd
path= r"D:\5th Semester\Artificial Neural Networks and Deep Learning\Labs\dataset\Image28.csv"

dataset= pd.read_csv(path)
pixels=dataset.iloc[:0]

img=pixels.astype("float32")
