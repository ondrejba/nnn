import pickle
import numpy as np

import classes.data as data
from classes.cost import SoftmaxCrossEntropy
from classes.layer import BatchNorm, LinearLayer, ReLU, Softmax, Convolution2D, Flatten
from classes.network import FeedForward
from classes.schedule import ExponentialSchedule

PICKLE_FILE = "datasets/MNIST/MNIST.pickle"

# load dataset
with open(PICKLE_FILE, "rb") as file:
  dataset = pickle.load(file)

# parameters
learning_rate = 1
batch_size = 32
num_steps = 10000
enable_schedule = True
report_test_accuracy = True

# prep. images for a dense network
dataset["train_data"] = np.expand_dims(dataset["train_data"], axis=-1) / 255
dataset["valid_data"] = np.expand_dims(dataset["valid_data"], axis=-1) / 255
dataset["test_data"] = np.expand_dims(dataset["test_data"], axis=-1) / 255

feed = data.BatchFeed(dataset["train_data"], dataset["train_labels"], batch_size=batch_size)

layers = [
  Convolution2D([28, 28, 1], [7, 7, 1], 16, [1, 1], "same", learning_rate),
  ReLU(),
  Convolution2D([28, 28, 16], [3, 3, 16], 16, [1, 1], "same", learning_rate),
  ReLU(),
  Convolution2D([28, 28, 16], [3, 3, 16], 16, [1, 1], "same", learning_rate),
  ReLU(),
  Flatten(),
  LinearLayer(12544, 10, learning_rate),
  Softmax()
]

cost = SoftmaxCrossEntropy()

network = FeedForward(layers, cost)

print("training a network with {:d} parameters".format(network.count_parameters()))

schedule = ExponentialSchedule(learning_rate, 100, 0.9)

costs = []

for i in range(num_steps):

  input_data, labels = feed.get_batch()

  forward = network.feed_forward(input_data)
  costs.append(network.compute_cost(forward, labels))
  network.backprop(forward, labels)

  if enable_schedule and i != 0 and i % 10 == 0:
    network.update_learning_rate(schedule.get_learning_rate(i))

  if i != 0 and i % 1000 == 0:
    print("step {:d}".format(i))
    print("cost 100 steps average: {:.6f}".format(np.mean(costs[len(costs) - 100:])))

    forward = network.feed_forward(dataset["valid_data"])
    valid_accuracy = data.one_hot_accuracy(forward, dataset["valid_labels"])

    print("validation accuracy: {:.2f}%".format(valid_accuracy * 100))
    print()

if report_test_accuracy:
  forward = network.feed_forward(dataset["test_data"])
  test_accuracy = data.one_hot_accuracy(forward, dataset["test_labels"])

  print("test accuracy: {:.2f}%".format(test_accuracy * 100))
  print()