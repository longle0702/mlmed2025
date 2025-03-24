import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd

df = pd.read_csv(r'D:\USTH\Nam Ba\ml in med\mlmed2025\Labwork 2\training_set_pixel_size_and_HC.csv')
X = df[['pixel size(mm)']]
y = df['head circumference (mm)']

plt.scatter(X, y)
plt.xlabel('pixel size(mm)')
plt.ylabel('head circumference (mm)')
plt.title('Pixel Size vs Head Circumference')
plt.show()