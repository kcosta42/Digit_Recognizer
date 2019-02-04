import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

def display_data(X):
  m, n = X.shape
  size = int(np.sqrt(n))
  nrows = int(np.sqrt(m))
  ncols = int(m / nrows)

  width = ncols * size
  height = nrows * size

  array = -np.ones(width * height)

  k = 0
  for r in range(0, nrows):
    for c in range(0, ncols):
      if k > m:
        break

      for i in range(0, n, size):
        index = (c * size) + int(i / size) * width + r * ncols * size * size
        array[index:index + size] = X[k][i:i + size]
      k = k + 1

    if k > m:
      break

  array = array.reshape(width, height)

  plt.imshow(array, cmap='gray', extent=[-1, 1, -1, 1])
  plt.show()
