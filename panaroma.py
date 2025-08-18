import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imageio
import warnings

cv.ocl.setUseOpenCL(0)
warnings.filterwarnings('ignore')

feature_extraction_algo = 'sift'
feature_to_match = 'bf'

train_photo = cv.imread('photos/train.jpg')
train_photo = cv.cvtColor(train_photo, cv.COLOR_BGR2RGB)
train_photo_gray = cv.cvtColor(train_photo, cv.COLOR_RGB2GRAY)

query_photo = cv.imread('photos/query.jpg')
query_photo = cv.cvtColor(query_photo, cv.COLOR_BGR2RGB)
query_photo_gray = cv.cvtColor(query_photo, cv.COLOR_RGB2GRAY)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout = False, figsize=(16, 9))
ax1.imshow(query_photo, cmap='gray')
ax1.set_xlabel('Query Photo', fontsize=14)

ax2.imshow(train_photo, cmap='gray')
ax2.set_xlabel('Train Photo', fontsize=14)
