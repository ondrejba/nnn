import numpy as np

class SoftmaxCrossEntropy(object):

  def __init__(self):
    pass

  def compute_cost(self, data, labels):
    n_samples = data.shape[0]

    # prevent underflow
    data = np.clip(data, 1e-7, 1.0 - 1e-7)

    return -(1.0 / n_samples) * np.sum(np.multiply(labels, np.log(data)))

  def compute_grad(self, data, labels):
    return data - labels