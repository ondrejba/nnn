from abc import ABC, abstractclassmethod
import numpy as np
import scipy.stats as stats

class Layer(ABC):

  @abstractclassmethod
  def feed_forward(self, data):
    """ Compute the feedforward activations. """
    pass

  @abstractclassmethod
  def backprop(self, grad_output):
    """ Compute a gradient w.r.t. inputs and a gradient w.r.t. parameters. """
    pass

  def count_parameters(self):
    return 0

class LinearLayer(Layer):

  def __init__(self, n_inputs, n_outputs, learning_rate):
    self.learning_rate = learning_rate

    # calculate parameters for weight initialization
    normal_stddev = 1.0 / np.sqrt(n_inputs)
    normal_mean = 0
    normal_min = -2 - normal_mean
    normal_max = 2 - normal_mean

    # initialize weights and biases
    self.weights = stats.truncnorm.rvs(normal_min, normal_max, loc=normal_mean, scale=normal_stddev, size=(n_inputs, n_outputs))
    self.biases = np.zeros((1, n_outputs))

    # initialize other variables
    self.input = None
    self.grad_weights = None
    self.grad_bias = None

  def feed_forward(self, data):
    self.input = data
    return np.matmul(data, self.weights) + self.biases

  def backprop(self, grad_output):
    self.grad_weights = np.matmul(np.transpose(self.input), grad_output)
    self.grad_bias = np.mean(grad_output, axis=0)
    grad_input = np.matmul(grad_output, np.transpose(self.weights))

    self.update(self.learning_rate)
    return grad_input

  def update(self, learning_rate):
    self.weights -= learning_rate * self.grad_weights / self.input.shape[0]
    self.biases -= learning_rate * self.grad_bias / self.input.shape[0]

  def count_parameters(self):
    count = 0
    count += self.weights.shape[0] * self.weights.shape[1]
    count += self.biases.shape[1]
    return count

class ReLU(Layer):

  def __init__(self):
    self.input = None

  def feed_forward(self, data):
    self.input = data
    return np.maximum(0, data)

  def backprop(self, grad_output):
    return np.multiply(grad_output, (self.input > 0).astype(np.int32))

class Softmax(Layer):

  def __init__(self):
    self.input = None
    self.output = None

  def feed_forward(self, data):

    data = np.clip(data, -500, 500)

    norm_sum = np.reshape(np.sum(np.exp(data), axis=1), (data.shape[0], 1))
    self.output = np.exp(data) / norm_sum

    return self.output

  def backprop(self, grad_output):
    """ Gradient already calculated by softmax cross entropy. """
    return grad_output
