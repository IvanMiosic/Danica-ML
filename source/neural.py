import theano
import numpy as np
from api_for_data_upload import open_images, convert_image, slice_image, read_locations
import sys
sys.path.append('../DeepLearningPython35')
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, ReLU, dropout_layer, SoftmaxLayer
from theano.tensor.nnet import softmax
import theano.tensor as T
import glob
from PIL import Image, ImageFilter
import os

def fix_axes(img):
    img = np.transpose(img, (2, 0, 1))
    return img.flatten()

def collect_all(folder):
    imgs = open_images(folder)
    regions = []
    for img in imgs: regions += slice_image(img)
    for i in range(len(regions)):
        regions[i]["img"] = fix_axes(convert_image(regions[i]["img"]))
    inputs = [region["img"] for region in regions]
    inputs = theano.shared(np.asarray(inputs, dtype=theano.config.floatX), borrow=True)
    labels = [region["count"] for region in regions]
    labels = theano.shared(np.asarray(labels, dtype=theano.config.floatX), borrow=True)
    return (inputs, theano.tensor.cast(labels, "int32"))

training = collect_all("../images/training_images")
testing = collect_all("../images/test_images")

print("loaded")

H, W = 85, 64
HH, WW = (H-4)//2, (W-4)//2
HHH, WWW = (HH-4)//2, (WW-4)//2
mini_batch_size = 32



class FormalityLayer(object):

    def __init__(self):
        self.n_in = n_in = 1
        self.n_out = n_out = 1
        self.activation_fn = ReLU
        self.p_dropout = 0.0
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b).reshape((mini_batch_size,))
        # self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(abs(y - self.output))

    def cost(self, net):
        return T.mean((net.y - self.output)**2)

    def otpt(self): return self.output
    
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 3, H, W),
                  filter_shape=(10, 3, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 10, HH, WW),
                  filter_shape=(10, 10, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=10*HHH*WWW, n_out=1, activation_fn=ReLU),
    # FullyConnectedLayer(n_in=100, n_out=1, activation_fn=ReLU),
    FormalityLayer()
    #SoftmaxLayer(n_in=100, n_out=10)
], mini_batch_size)
net.SGD(training, 1, mini_batch_size, 0.03, testing, testing, lmbda=0.1)
