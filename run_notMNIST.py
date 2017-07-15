from six.moves import cPickle as pickle
from classes.layer import LinearLayer, ReLU, Softmax
from classes.cost import SoftmaxCrossEntropy
from classes.network import FeedForward
import numpy as np

pickle_file = '../learn_tensorflow/notMNIST'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)



image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# parameters
learning_rate = 0.01
batch_size = 128
num_steps = 10000

layers = [
  LinearLayer(784, 1024, learning_rate),
  ReLU(),
  LinearLayer(1024, 512, learning_rate),
  ReLU(),
  LinearLayer(512, 10, learning_rate),
  Softmax()
]

cost = SoftmaxCrossEntropy()

network = FeedForward(layers, cost)

def get_batch(index):
  start = batch_size * index % len(train_dataset)
  end = start + batch_size

  return train_dataset[start:end]

for i in range(num_steps):
  forward = network.feed_forward(train_dataset)
  c = network.compute_cost(forward, train_labels)

  if i % 100 == 0:
    print('Cost: %f' % c)

  network.backprop(forward, train_labels)