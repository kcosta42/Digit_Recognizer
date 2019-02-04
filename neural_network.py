import numpy as np

class NeuralNetwork(object):
  """Neural Network

  Parameters
  ----------
  shape: 1d-array
    Number of units in each layers
  eta: float
    Learning rate (between 0.0 and 1.0)
  n_iter: int
    Passes over the training dataset
  Lambda: float
    Lambda term for regularization
  epsilon: float
    Epsilon term for computing numerical gradient

  Attributes
  ----------
  _w = array-like
    Weigths after fitting
  _cost: list
    Number of cost values
  """
  def __init__(self, shape, eta=0.1, n_iter=50, Lambda=0, epsilon=1e-4):
    self.shape = shape
    self.eta = eta
    self.n_iter = n_iter
    self.Lambda = Lambda
    self.epsilon = epsilon
    self._w = {}
    self._cost = []

  def cost_function(self, nn_params, X, y):
    L = len(self.shape) - 1

    for l in range(0, L):
      shape = (self.shape[l + 1], (self.shape[l] + 1))
      start = self.shape[l] * (self.shape[l - 1] + 1) if l > 0 else 0
      end = start + shape[0] * shape[1]
      self._w[l] = np.reshape(nn_params[start:end], shape)

    m = np.size(X, 0)
    K = self.shape[-1]

    yVec = np.zeros((K, y.shape[0]))
    for i in range(0, y.shape[0]):
      yVec[int(y[i] - 1)][i] = 1

    # Forward Propagation
    Theta = {}
    z = {}
    a = {}
    z[0] = np.nan
    a[0] = np.insert(X, 0, 1, axis=1)
    for l in range(0, L):
      Theta[l] = self._w[l]
      z[l + 1] = Theta[l].dot(a[l].T).T
      a[l + 1] = self.sigmoid(z[l + 1])
      if (l + 1 < L):
        a[l + 1] = np.insert(a[l + 1], 0, 1, axis=1)

    lhs = sum(sum(-yVec * np.log(a[L].T)))
    rhs = sum(sum((1 - yVec) * np.log(1 - a[L].T)))

    regularization = 0
    for l in range(0, L):
      regularization = regularization + sum(sum(Theta[l][:, 1:] ** 2))

    J = (1 / m) * (lhs - rhs) + (self.Lambda / (2 * m)) * regularization
    self._cost.append(J)

    # Back Propagation
    DELTA = {}
    delta = {}
    delta[L] = a[L].T - yVec
    for l in range(L - 1, 0, -1):
      delta[l] = Theta[l][:, 1:].T.dot(delta[l + 1]) * self.sigmoid_gradient(z[l]).T
    for l in range(0, L):
      DELTA[l] = delta[l + 1].dot(a[l])

    grad = np.array([])
    for l in range(0, L):
      regularization = (self.Lambda / m) * (Theta[l][:, 1:])
      Theta_grad = (1 / m) * DELTA[l] + np.insert(regularization, 0, 0, axis=1)

      grad = np.append(grad, Theta_grad.flatten())

    return (J, grad)

  def compute_numerical_gradient(self, J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)

    for p in range(0, np.size(theta)):
      perturb[p] = self.epsilon
      loss1, dummy = J(theta - perturb)
      loss2, dummy = J(theta + perturb)
      numgrad[p] = (loss2 - loss1) / (2 * self.epsilon)
      perturb[p] = 0
    return numgrad

  def rand_initialize_weights(self, L_in, L_out):
    epsilon = 0.12
    W = np.random.rand(L_out, 1 + L_in) * (2 * epsilon) - epsilon
    return W

  def predict(self, nn_params, X):
    L = len(self.shape) - 1
    p = np.zeros((X.shape[0], 1))

    Theta = {}
    for l in range(0, L):
      shape = (self.shape[l + 1], (self.shape[l] + 1))
      start = self.shape[l] * (self.shape[l - 1] + 1) if l > 0 else 0
      end = start + shape[0] * shape[1]
      Theta[l] = np.reshape(nn_params[start:end], shape)

    prev = self.sigmoid(np.insert(X, 0, 1, axis=1).dot(Theta[0].T))
    for l in range(1, L):
      prediction = self.sigmoid(np.insert(prev, 0, 1, axis=1).dot(Theta[l].T))
      prev = prediction

    p = np.array([x + 1 for x in prediction.argmax(1)])
    return p

  def sigmoid(self, z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g

  def sigmoid_gradient(self, z):
    g = self.sigmoid(z) * (1 - self.sigmoid(z))
    return g
