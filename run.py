from layer import LinearLayer, ReLU, Softmax
from cost import SoftmaxCrossEntropy
from network import FeedForward
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

data_1 = np.random.normal(loc=5, scale=1, size=(20, 2))
data_2 = np.random.normal(loc=-5.0, scale=1.5, size=(20, 2))
data_3 = np.random.normal(loc= 10.0, scale=0.5, size=(20, 2))

labels_1 = np.zeros((20, 3))
labels_2 = np.zeros((20, 3))
labels_3 = np.zeros((20, 3))

for i in range(20):
  labels_1[i, 0] = 1
  labels_2[i, 1] = 1
  labels_3[i, 2] = 1

data = np.concatenate((data_1, data_2, data_3))
labels = np.concatenate((labels_1, labels_2, labels_3))

data, labels = shuffle(data, labels)

layers = [
  LinearLayer(2, 20),
  ReLU(),
  LinearLayer(20, 20),
  ReLU(),
  LinearLayer(20, 20),
  ReLU(),
  LinearLayer(20, 3),
  Softmax()
]

cost = SoftmaxCrossEntropy()

network = FeedForward(layers, cost)

for i in range(100000):
  forward = network.feed_forward(data)
  c = network.compute_cost(forward, labels)
  if math.isnan(c):
    break

  if i % 1000 == 0:
    print('Cost: %f' % c)
  network.backprop(forward, labels)