import pickle
import numpy as np

from classes.cost import SoftmaxCrossEntropy
from classes.data import Dataset
from classes.layer import LinearLayer, ReLU, Softmax
from classes.network import FeedForward

PICKLE_FILE = "datasets/MNIST/MNIST.pickle"

# load dataset
with open(PICKLE_FILE, "rb") as file:
  dataset = pickle.load(file)

# parameters
learning_rate = 0.001
batch_size = 32
num_steps = 10000000

# prep. images for a dense network
dataset["train_data"] = np.reshape(dataset["train_data"], (-1, 28 * 28)) / 255
dataset["valid_data"] = np.reshape(dataset["train_data"], (-1, 28 * 28)) / 255
dataset["test_data"] = np.reshape(dataset["train_data"], (-1, 28 * 28)) / 255

feed = Dataset(dataset["train_data"], dataset["train_labels"], dataset["valid_data"], dataset["valid_labels"], batch_size=batch_size)

layers = [
  LinearLayer(784, 32, learning_rate),
  ReLU(),
  LinearLayer(32, 10, learning_rate),
  Softmax()
]

cost = SoftmaxCrossEntropy()

network = FeedForward(layers, cost)

costs = []

for i in range(num_steps):

  data, labels = feed.get_batch()

  forward = network.feed_forward(data)
  costs.append(network.compute_cost(forward, labels))

  if i != 0 and i % 100 == 0:
    print(i)
    print("cost 100 steps average: {:.6f}".format(np.mean(costs[len(costs) - 100:])))

  network.backprop(forward, labels)
