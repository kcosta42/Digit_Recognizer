from display_data import display_data
from neural_network import NeuralNetwork
from paint import Paint

import pandas as pd
import numpy as np

from scipy import optimize

X = np.array([])
y = np.array([])

def addTrainingExample(x, pred):
  global X, y

  X = np.insert(X, 0, x, axis=0) if X.any() else np.array(x)
  y = np.insert(y, 0, pred, axis=0) if y.any() else np.array(pred)
  print(y.shape)

def main():
  global X, y

  df = pd.read_csv("./datasets/digits.csv", header=None)
  X = df.values
  df = pd.read_csv("./datasets/labels.csv", header=None)
  y = df.values

  sel = np.random.permutation(X)
  sel = sel[0:100]

  display_data(sel)

  input_layer_size = 400
  hidden_layer_size = 25
  num_labels = 10

  shape = (input_layer_size, hidden_layer_size, num_labels)

  for i in range(1, 11):
    prevLen = len(y)

    print(f'Write example for {i if i < 10 else 0}')
    f = lambda x: addTrainingExample(x, [[i]])
    Paint(func=f)

    if len(y) == prevLen:
      break

  nn = NeuralNetwork(shape)
  nn.Lambda = 1

  initial_Theta1 = nn.rand_initialize_weights(input_layer_size, hidden_layer_size)
  initial_Theta2 = nn.rand_initialize_weights(hidden_layer_size, num_labels)
  initial_nn_params = np.array(initial_Theta1.flatten().tolist() + initial_Theta2.flatten().tolist())

  f = lambda p : nn.cost_function(p, X, y)[0]
  gradf = lambda p : nn.cost_function(p, X, y)[1]
  print('Training Neural Network... (This can take a while)')
  nn_params = optimize.fmin_cg(f, initial_nn_params, fprime=gradf, maxiter=250)

  shape = hidden_layer_size, (input_layer_size + 1)
  idx = hidden_layer_size * (input_layer_size + 1)
  Theta1 = np.reshape(nn_params[0:idx], shape)

  display_data(Theta1[:, 1:])

  pred = nn.predict(nn_params, X)
  print(f'Training Set Accuracy: {np.mean(pred == y.T) * 100}%')

  p = lambda x: nn.predict(nn_params, x)
  Paint(predict=p)

  file = open("./datasets/digits.csv", 'w+')
  for i in range(len(X)):
    file.write(f'{", ".join(str(x) for x in X[i])}\n')
  file.close()

  file = open("./datasets/labels.csv", 'w+')
  for l in y:
    file.write(f'{l[0]}\n')
  file.close()

if __name__ == "__main__":
  main()
