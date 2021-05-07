import numpy as np

def one_hot(y, k):
    ONE_HOT = np.zeros((k, np.shape(y)[0]))
    for i, e in enumerate(y):
        ONE_HOT[e, i] = 1
    return ONE_HOT

def split_into_batches(batch_size, X, Y_one_hot):
    n_batches = len(X.T) // batch_size
    X_batches = np.hsplit(X, n_batches)
    Y_batches = np.hsplit(Y_one_hot, n_batches)
    return X_batches, Y_batches

def softmax(Z):
    s = np.exp(Z - np.max(Z))
    a = s / np.sum(s, axis=0)
    return a


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0) * 1.0


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_derivative(Z):
    f = sigmoid(Z)
    return f * (1 - f)


def softmax(Z):
    e = np.exp(Z)
    return e / np.sum(e, axis=0, keepdims=True)


def softmax_derivative(Z):
    A = softmax(Z)
    out = np.zeros_like(A)
    for example in A.T:
        example = example[:, None]
        out += example @ np.ones_like(example).T * ((np.identity(np.shape(example)[0])) - A.T)
    return


class ActivationFunction:
    def __init__(self, activation, derivative):
        self.activation = activation
        self.derivative = derivative
