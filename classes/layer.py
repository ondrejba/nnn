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

  def __init__(self, input_shape, filter_shape, num_filters, stride, padding, learning_rate):
    self.input_shape = input_shape
    self.filter_shape = filter_shape
    self.stride = stride
    self.num_filters = num_filters
    self.padding = padding
    self.learning_rate = learning_rate

    # calculate parameters for weight initialization
    normal_stddev = 1.0 / np.sqrt(np.product(input_shape))
    normal_mean = 0
    normal_min = -2 - normal_mean
    normal_max = 2 - normal_mean

    self.weights = stats.truncnorm.rvs(normal_min, normal_max, loc=normal_mean, scale=normal_stddev, size=(*self.filter_shape, num_filters))
    self.biases = np.zeros((1, num_filters))

    self.input = None
    self.grad_weights = None
    self.grad_bias = None

  def feed_forward(self, data):
    self.input = data
    return self.__convolve(self.input, self.weights, self.padding) + self.biases

  def backprop(self, grad_output):

    weights_transpose = np.transpose(self.weights, axes=[0, 2, 1, 3])
    grad_transpose = np.transpose(grad_output, axes=[0, 2, 1, 3])

    self.grad_weights = self.__convolve_deltas(grad_transpose, self.input, mode=self.padding)
    self.grad_weights = np.mean(self.grad_weights, axis=0)

    self.grad_biases = np.mean(np.sum(grad_output, axis=(1, 2, 3)), axis=0) 

    grad_outputs = self.__convolve(self.input, weights_transpose, mode=self.padding)

    self.update(self.learning_rate)

    return grad_outputs

  def update(self, learning_rate):

    print(self.grad_weights.shape)
    print(self.grad_biases.shape)
    self.weights -= learning_rate * self.grad_weights
    self.biases -= learning_rate * self.grad_biases

  def __convolve(self, inputs, filters, mode):

    print(inputs.shape)
    print(filters.shape)

    output_stack = []

    for output_filter_idx in range(filters.shape[3]):

      batch_stack = []

      for batch_index in range(inputs.shape[0]):

        image_response = []

        for input_filter_idx in range(filters.shape[2]):

          out = signal.convolve2d(inputs[batch_index, :, :, input_filter_idx], 
                                  filters[:, :, input_filter_idx, output_filter_idx], mode=mode)

          image_response.append(out)

        # stack depth-wise
        image_response = np.sum(image_response, axis=0)
        batch_stack.append(image_response)

      # stack batch-wise
      batch_stack = np.stack(batch_stack, axis=0)
      output_stack.append(batch_stack)

    # stack depth-wise
    output_stack = np.stack(output_stack, axis=-1)

    return output_stack

  def __convolve_deltas(self, deltas, inputs, mode):

    print(deltas.shape)
    print(inputs.shape)

    batch_stack = []

    for batch_index in range(deltas.shape[0]):

      image_response = []

      for input_filter_idx in range(deltas.shape[3]):

        out = signal.convolve2d(deltas[batch_index, :, :, input_filter_idx], 
                                inputs[batch_index, :, :, input_filter_idx], mode=mode)

        image_response.append(out)

      # stack depth-wise
      image_response = np.sum(image_response, axis=0)
      batch_stack.append(image_response)

    # stack batch-wise
    batch_stack = np.stack(batch_stack, axis=0)

    return batch_stack

class BatchNorm(Layer):

  def __init__(self, num_filters, convolutional=False, update_moving_mean_and_var=True):

    # hyper parameters
    self.epsilon = 1e-4
    self.alpha = 0.999
    self.convolutional = convolutional
    self.update_moving_mean_and_var = update_moving_mean_and_var

    # trainable parameters
    self.gamma = np.ones(num_filters)
    self.beta = np.zeros(num_filters)

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

class Flatten(Layer):

  def __init__(self):
    self.data_height = None
    self.data_width = None

  def feed_forward(self, data):
    self.data_height = data.shape[1]
    self.data_width = data.shape[2]
    self.data_depth = data.shape[3]

    return np.reshape(data, [data.shape[0], self.data_height * self.data_width * self.data_depth])

  def backprop(self, grad_output):
    return np.reshape(grad_output, [grad_output.shape[0], self.data_height, self.data_width, self.data_depth])

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


