import numpy as np
from api_for_data_upload import open_images, convert_image, slice_image, read_locations
import sys
import glob
from PIL import Image, ImageFilter
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
import time

W, H = 85, 64
mini_batch_size = 32

cnn = keras.models.load_model('cnn.h5')

avg_rel = 0
cnt = 0
for img in open_images("../images/test_images"):
    # predictions = cnn.predict(x=np.array([region["img"] for region in img]), batch_size=mini_batch_size) * 5
    # prediction = 0
    # for p in predictions: prediction += p
    # actual = 0
    # for region in img: actual += region["count"]
    # print(prediction, actual, abs(prediction/actual - 1))
    print(img["name"])
    width = img["img"].size[0]
    height = img["img"].size[1]
    prediction = 0
    args = []
    for i in range(0, width-W+1, 15):
        for j in range(0, height-H+1, 15):
            args.append([np.asarray(img["img"].crop([i, j, i+W, j+H]))/255])
    predictions = cnn.predict(x=np.array(args).reshape((len(args), H, W, 3)), batch_size=mini_batch_size)
    for p in predictions: prediction += p
    prediction *= (width/W) * (height/H) / len(args)
    actual = len(img["loc"]["X"])
    print(img["name"], prediction, actual, abs(prediction/actual - 1))
    cnt += 1
    avg_rel += abs(prediction/actual - 1)
    
print(avg_rel/cnt)

cnn.summary()

cnn.save('cnn.h5')
