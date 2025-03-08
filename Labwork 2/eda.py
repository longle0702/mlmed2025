import matplotlib.pyplot as plt
from PIL import Image
import os

gt = r'D:\USTH\Nam Ba\ml in med\mlmed2025\Labwork 2\1\val_set\648_HC.png'
mask = r'D:\USTH\Nam Ba\ml in med\mlmed2025\Labwork 2\1\val_mask\648_HC_Annotation.png'
plt.figure(figsize=(12, 10))
plt.subplot(1, 2, 1)
plt.imshow(Image.open(gt), cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(Image.open(mask), cmap='gray')
plt.title('Ground Truth')
plt.axis('off')
plt.tight_layout()