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

def collect_all(folder):
    imgs = open_images(folder)
    outputs = []
    for img in imgs: outputs.append(slice_image(img))
    for i in range(len(outputs)):
        for j in range(len(outputs[i])):
            outputs[i][j]["img"] = convert_image(outputs[i][j]["img"])
    return outputs

training = collect_all("../images/training_images")
# testing = collect_all("../images/test_images")

print("loaded")

W, H = 85, 64
mini_batch_size = 32

cnn = Sequential()

cnn.add(Conv2D(filters=20, 
               kernel_size=(3,3), 
               padding='valid',
               input_shape=(H,W,3),
               use_bias=True,
               data_format='channels_last'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     border_mode='valid'))

cnn.add(Conv2D(filters=30,
               kernel_size=(5,5),
               use_bias=True,
               padding='valid'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     border_mode='valid'))

cnn.add(Conv2D(filters=40,
               kernel_size=(6,6),
               use_bias=True,
               padding='valid'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     border_mode='valid'))

cnn.add(Flatten())

cnn.add(Dense(32, use_bias=True))
cnn.add(Activation('relu'))
cnn.add(Dropout(0.25))

cnn.add(Dense(1))
cnn.add(Activation('linear'))

optimizer = keras.optimizers.Adam()
cnn.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])

start = time.time()
cnn.fit(
    x = np.array([region["img"] for regions in training for region in regions]),
    y = np.array([region["count"] for regions in training for region in regions]),
    batch_size = mini_batch_size,
    epochs = 5,
    validation_split = 0.2
)
end = time.time()
print('Processing time:',(end - start)/60)
# cnn.save_weights('cnn_baseline.h5')

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
