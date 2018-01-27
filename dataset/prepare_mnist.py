import os, pickle
import numpy as np
from scipy.ndimage import imread
from sklearn.utils import shuffle

VALIDATION_FRACTION = 0.1

def load_image(path):
  return imread(path)

def load_digit(root, digit):
  path = os.path.join(root, str(digit))
  images = []

  for img in os.listdir(path):
    img_path = os.path.join(path, img)
    images.append(load_image(img_path))

  label = [0] * 10
  label[digit] = 1

  return images, [label] * len(images)

data = []
labels = []

test_data = []
test_labels = []

for digit in range(10):
  digit_data, digit_lables = load_digit("training", digit)
  digit_data_test, digit_labels_test = load_digit("testing", digit)

  data += digit_data
  labels += digit_lables

  test_data += digit_data_test
  test_labels += digit_labels_test

data = np.array(data)
labels = np.array(labels)

data, labels = shuffle(data, labels)

dset_size = data.shape[0]

valid_size = int(VALIDATION_FRACTION * dset_size)
train_size = dset_size - valid_size

train_data = data[:train_size]
train_labels = labels[:train_size]
#train_labels = np.expand_dims(train_labels, axis=1)

valid_data = data[train_size:]
valid_labels = labels[train_size:]
#valid_labels = np.expand_dims(valid_labels, axis=1)

test_data = np.array(test_data)
test_labels = np.array(test_labels)
#test_labels = np.expand_dims(test_labels, axis=1)

print("Training:")
print(train_data.shape)
print(train_labels.shape)

print("Validation:")
print(valid_data.shape)
print(valid_labels.shape)

print("Testing:")
print(test_data.shape)
print(test_labels.shape)

with open("MNIST.pickle", "wb") as file:
  pickle.dump({
    "train_data": train_data,
    "train_labels": train_labels,
    "valid_data": valid_data,
    "valid_labels": valid_labels,
    "test_data": test_data,
    "test_labels": test_labels
  }, file)