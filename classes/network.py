

class FeedForward(object):

  def __init__(self, layers, cost):
    self.layers = layers
    self.cost = cost

  def feed_forward(self, data):
    next_input = data

    for layer in self.layers:
      next_input = layer.feed_forward(next_input)

    return next_input

  def backprop(self, data, labels):
    next_grad = self.cost.compute_grad(data, labels)

    for layer in reversed(self.layers):
      next_grad = layer.backprop(next_grad)

  def compute_cost(self, data, labels):
    return self.cost.compute_cost(data, labels)