
class DiscreteSchedule:

  def __init__(self, marks, learning_rates):
    self.marks = marks
    self.learning_rates = learning_rates

  def get_learning_rate(self, step):
    for i, mark in enumerate(reversed(self.marks)):
      if step >= mark:
        return self.learning_rates[len(self.learning_rates) - i - 1]