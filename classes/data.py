import numpy as np

class BatchFeed():
  def __init__(self, train_data, train_labels, batch_size=32):
    self.train_data = train_data
    self.train_labels = train_labels

    self.i = 0
    self.batch_size = batch_size
    self.size = self.train_data.shape[0]

  def get_batch(self):
    ind = range(self.i, self.i + self.batch_size)

    data = self.train_data.take(ind, axis=0, mode="wrap")
    labels = self.train_labels.take(ind, axis=0, mode="wrap")

    self.i = (self.i + self.batch_size) % self.size

    return data, labels

def one_hot_accuracy(predictions, labels):

  arg_predictions = np.argmax(predictions, axis=1)
  arg_labels = np.argmax(labels, axis=1)

  correct = np.sum(np.equal(arg_labels, arg_predictions))
  total = labels.shape[0]

  return correct / total