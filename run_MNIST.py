import pickle
import numpy as np

import classes.data as data
from classes.cost import SoftmaxCrossEntropy
from classes.layer import BatchNorm, LinearLayer, ReLU, Softmax
from classes.network import FeedForward
from classes.schedule import DiscreteSchedule

PICKLE_FILE = "datasets/MNIST/MNIST.pickle"

# load dataset
with open(PICKLE_FILE, "rb") as file:
  dataset = pickle.load(file)

# parameters
learning_rate = 0.01
batch_size = 32
num_steps = 100000
enable_schedule = True
report_test_accuracy = True

# prep. images for a dense network
dataset["train_data"] = np.reshape(dataset["train_data"], (-1, 28 * 28)) / 255
dataset["valid_data"] = np.reshape(dataset["valid_data"], (-1, 28 * 28)) / 255
dataset["test_data"] = np.reshape(dataset["test_data"], (-1, 28 * 28)) / 255

feed = data.BatchFeed(dataset["train_data"], dataset["train_labels"], batch_size=batch_size)

layers = [
  LinearLayer(784, 512, learning_rate),
  BatchNorm(512),
  ReLU(),
  LinearLayer(512, 256, learning_rate),
  BatchNorm(256),
  ReLU(),
  LinearLayer(256, 128, learning_rate),
  BatchNorm(128),
  ReLU(),
  LinearLayer(128, 64, learning_rate),
  BatchNorm(64),
  ReLU(),
  LinearLayer(64, 10, learning_rate),
  Softmax()
]

cost = SoftmaxCrossEntropy()

network = FeedForward(layers, cost)

print("training a network with {:d} parameters".format(network.count_parameters()))

schedule = DiscreteSchedule([0, 30000, 60000], [0.01, 0.005, 0.001])

costs = []

for i in range(num_steps):

  input_data, labels = feed.get_batch()

  forward = network.feed_forward(input_data)
  costs.append(network.compute_cost(forward, labels))
  network.backprop(forward, labels)

  if i != 0 and i % 1000 == 0:
    print("step {:d}".format(i))
    print("cost 100 steps average: {:.6f}".format(np.mean(costs[len(costs) - 100:])))

    forward = network.feed_forward(dataset["valid_data"])
    valid_accuracy = data.one_hot_accuracy(forward, dataset["valid_labels"])

    print("validation accuracy: {:.2f}%".format(valid_accuracy * 100))
    print()

    if enable_schedule:
      network.update_learning_rate(schedule.get_learning_rate(i))

if report_test_accuracy:
  forward = network.feed_forward(dataset["test_data"])
  test_accuracy = data.one_hot_accuracy(forward, dataset["test_labels"])

  print("test accuracy: {:.2f}%".format(test_accuracy * 100))
  print()