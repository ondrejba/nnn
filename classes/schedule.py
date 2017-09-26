
class DiscreteSchedule:

  def __init__(self, marks, learning_rates):
    self.marks = marks
    self.learning_rates = learning_rates

  def get_learning_rate(self, step):
    for i, mark in enumerate(reversed(self.marks)):
      if step >= mark:
        return self.learning_rates[len(self.learning_rates) - i - 1]

class ExponentialSchedule:

  def __init__(self, learning_rate, decay_steps, decay_rate):
    self.learning_rate = learning_rate
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate

  def get_learning_rate(self, step):
    return self.learning_rate * (self.decay_rate ** (step / self.decay_steps))