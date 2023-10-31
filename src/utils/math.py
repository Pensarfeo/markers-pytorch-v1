class MovingAverage():
  def __init__(self, n):
    self.n = n
    self.cache = [0]*n
    self.i = 0
    
  def add(self, v):
    p = self.i%(self.n)
    self.cache[p] = v
    self.result = 0
    for v in self.cache:
      self.result += v
    self.result /= self.n
    self.i += 1
    return self.result
