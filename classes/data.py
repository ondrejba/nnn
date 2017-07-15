class Dataset():
  def __init__(self, train_data, train_labels, valid_data, valid_labels, batch_size=32):
    self.train_data = train_data
    self.train_labels = train_labels
    self.valid_data = valid_data
    self.valid_labels = valid_labels

    self.i = 0
    self.batch_size = batch_size
    self.size = self.train_data.shape[0]

  def get_batch(self):
    ind = range(self.i, self.i + self.batch_size)

    data = self.train_data.take(ind, axis=0, mode="wrap")
    labels = self.train_labels.take(ind, axis=0, mode="wrap")

    self.i = (self.i + self.batch_size) % self.size

    return data, labels

  def get_valid_data(self):
    return self.valid_data, self.valid_labels