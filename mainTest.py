import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumorDetection.keras')

image=cv2.imread('D://BrainTumor//BrainTumorDetection//pred//pred0.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=model.predict(input_img)

tumor_probability = result[0][1]
tumor_probability = int(tumor_probability)
print(tumor_probability)