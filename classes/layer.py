from abc import ABC, abstractclassmethod
import numpy as np
import scipy.stats as stats
import scipy.signal as signal

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

class Convolution2D(Layer):

  def __init__(self, input_shape, filter_shape, stride, num_filters, padding):
    self.input_shape = input_shape
    self.filter_shape = filter_shape
    self.stride = stride
    self.num_filters = num_filters
    self.padding = padding

    # calculate parameters for weight initialization
    normal_stddev = 1.0 / np.sqrt(np.product(input_shape))
    normal_mean = 0
    normal_min = -2 - normal_mean
    normal_max = 2 - normal_mean

    self.weights = stats.truncnorm.rvs(normal_min, normal_max, loc=normal_mean, scale=normal_stddev, size=(num_filters, *input_shape))
    self.biases = np.zeros((1, num_filters))

    self.input = None
    self.grad_weights = None
    self.grad_bias = None

  def feed_forward(self, data):
    self.input = data
    return signal.convolve(self.input, self.weights, mode=self.padding)

  def backprop(self, grad_output):
    self.grad_weights = signal.convolve(self.input, grad_output)
    return signal.convolve(grad_output, np.flip(self.weights, axis=0))

class BatchNorm(Layer):

  def __init__(self, convolutional=False, update_moving_mean_and_var=True):

    # hyper parameters
    self.epsilon = 1e-4
    self.alpha = 0.999
    self.convolutional = convolutional
    self.update_moving_mean_and_var = update_moving_mean_and_var

    # trainable parameters
    self.gamma = 1
    self.beta = 0

    # parameters to update
    self.moving_mean = None
    self.moving_var = None

    # memory
    self.input = None
    self.output = None
    self.minibatch_mean = None
    self.minibatch_var = None


  def feed_forward(self, data):

    self.input = data

    if self.convolutional:
      self.minibatch_mean = np.mean(data, axis=(0, 1, 2))
      self.minibatch_var = np.var(data, axis=(0, 1, 2))
    else:
      self.minibatch_mean = np.mean(data, axis=0)
      self.minibatch_var = np.var(data, axis=0)

    normalize = (data - self.minibatch_mean) / np.sqrt(self.minibatch_var + self.epsilon)
    self.output = self.gamma * normalize + self.beta

    return self.output

  def backprop(self, grad_output):

    batch_size = self.input.shape[0]
    stable_std = np.sqrt(self.minibatch_var + self.epsilon)

    grad_normalize = grad_output * self.gamma

    if self.convolutional:
      grad_var = np.sum(grad_normalize * (self.input - self.minibatch_mean) * (-1 / 2) * np.power(self.minibatch_var + self.epsilon, -3 / 2), axis=(0, 1, 2))
    else:
      grad_var = np.sum(grad_normalize * (self.input - self.minibatch_mean) * (-1 / 2) * np.power(self.minibatch_var + self.epsilon, -3 / 2), axis=0)

    if self.convolutional:
      grad_mean = np.sum(grad_normalize * (-1 / stable_std), axis=(0, 1, 2)) + grad_var * (np.sum(-2 * (self.input - self.minibatch_mean), axis=(0, 1, 2)) / batch_size)
    else:
      grad_mean = np.sum(grad_normalize * (-1 / stable_std), axis=0) + grad_var * (np.sum(-2 * (self.input - self.minibatch_mean), axis=0) / batch_size)

    grad_input = grad_normalize * (1 / stable_std) + grad_var * (2 * (self.input - self.minibatch_mean) / batch_size) + grad_mean * (1 / batch_size)
    self.grad_gamma = np.sum(grad_output * grad_normalize)
    self.grad_beta = np.sum(grad_output)

    return grad_input

  def update(self, learning_rate):

    self.gamma -= learning_rate * self.grad_gamma
    self.beta -= learning_rate * self.grad_beta

    if self.update_moving_mean_and_var:

      if self.moving_mean is None:
        self.moving_mean = self.minibatch_mean
      else:
        self.moving_mean += self.alpha * (self.minibatch_mean - self.moving_mean)

      if self.moving_var is None:
        self.moving_var = self.minibatch_var
      else:
        self.moving_var += self.alpha * (self.minibatch_var - self.moving_var)

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


