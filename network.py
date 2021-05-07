import numpy as np
import pickle as pkl
from typing import List
from nn_utils import *
import csv
import matplotlib.pyplot as plt

class NormalInitializer:
    def __init__(self, mean, std_dev):
        self._mean = mean
        self._std_dev = std_dev

    def generateWeights(self, layerSizes: List[int]):
        return [np.random.normal(self._mean, self._std_dev, (layerSizes[i + 1], layerSizes[i])) for i in range(len(layerSizes) - 1)]

    def name(self):
        return 'Normal'

class UniformInitializer:
    def __init__(self, lowerBound, upperBound):
        self._lowerBound = lowerBound
        self._upperBound = upperBound

    def generateWeights(self, layerSizes: List[int]):
        return [np.random.uniform(self._lowerBound, self._upperBound, (layerSizes[i + 1], layerSizes[i])) for i in range(len(layerSizes) - 1)]

    def name(self):
        return 'Uniform'

class XavierInitializer:
    def generateWeights(self, layerSizes: List[int]):
        return [np.random.normal(0, np.sqrt(2/(layerSizes[i] + layerSizes[i+1])), (layerSizes[i + 1], layerSizes[i])) for i in range(len(layerSizes) - 1)]

    def name(self):
        return 'Xavier'

class HeInitializer:
    def __init__(self, mean, std_dev):
        self._mean = mean
        self._std_dev = std_dev

    def generateWeights(self, layerSizes: List[int]):
        return [np.random.normal(0, 1, (layerSizes[i + 1], layerSizes[i])) * np.sqrt(2) / np.sqrt(layerSizes[i]) for i in
                range(len(layerSizes) - 1)]

    def name(self):
        return 'He'

class DeepNetwork:
    def __init__(self, sizes: List[int], hidden_activation: ActivationFunction, last_activation: ActivationFunction, weightGenerator):
        self.sizes = sizes
        self.activations = [hidden_activation for i in range(len(sizes) - 2)]
        self.activations.append(last_activation)
        self.biases = [np.zeros((size, 1)) for size in sizes[1:]]
        self.weights = weightGenerator.generateWeights(sizes)
        self._As = []
        self._Zs = []
        self._W_grads = [np.zeros_like(w) for w in self.weights]
        self._b_grads = [np.zeros_like(b) for b in self.biases]


    def classify(self, X):
        raw_output = self._feedforward(X)
        Y_classified = self._predict(raw_output)
        return Y_classified, raw_output

    def _feedforward(self, X):
        A = X
        for (W, b, f) in zip(self.weights, self.biases, self.activations):
            Z = W @ A + b
            A = f.activation(Z)
        return A

    def loss(self, Y_train, Y_pred):
        n = np.shape(Y_train)[1]
        return np.sum((Y_train - Y_pred) ** 2) / n

    def _predict(self, Y_hat):
        return np.argmax(Y_hat, axis=0)

    def accuracy(self, Y_target, Y):
        Y = Y[:, None]
        return np.sum(Y_target == Y) / len(Y_target)

    def error(self, Y_target, Y):
        return 1 - self.accuracy(Y_target, Y)

    def _feedforward_train(self, X):
        self._reset_cache()
        A = X
        self._As.append(A)
        for (W, b, f) in zip(self.weights, self.biases, self.activations):
            Z = W @ A + b
            self._Zs.append(Z)
            A = f.activation(Z)
            self._As.append(A)
        return A

    def train(self, X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, epochs, batch_size, optimizer):
        X_batches, Y_batches = split_into_batches(batch_size, X_train, Y_train_one)
        train_losses = []
        val_losses = []
        val_errors = []
        optimizer.initialize(self)
        for i in range(epochs):
            for (X_batch, Y_batch) in zip(X_batches, Y_batches):
                optimizer.train(self, X_batch, Y_batch)
            Y_val_hat, raw_val_output = self.classify(X_val)
            Y_train_hat, raw_train_output = self.classify(X_train)
            train_loss = self.loss(Y_train_one, raw_train_output)
            val_loss = self.loss(Y_val_one, raw_val_output)
            error = self.error(Y_val, Y_val_hat)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_errors.append(error)
            print(f"Train loss: {train_loss}, Val loss: {val_loss}, Val error: {error}")
        return train_losses, val_losses, val_errors

    def _mse_derivative(self, Y_target, Y):
        n = np.shape(Y_target)[1]
        return (Y_target - Y) / n

    def _backpropagate(self, Y_batch):
        n_examples = Y_batch.shape[1]
        grad = self.activations[-1].derivative(self._Zs[-1])
        delta = self._mse_derivative(Y_batch, self._As[-1]) * grad
        self._W_grads[-1] = delta @ self._As[-2].T / n_examples
        self._b_grads[-1] = np.mean(delta, axis=1, keepdims=True)

        for l in range(2, len(self.sizes)):
            grad = self.activations[-l].derivative(self._Zs[-l])
            delta = (self.weights[-l + 1].T @ delta) * grad
            self._W_grads[-l] = delta @ self._As[-l-1].T / n_examples
            self._b_grads[-l] = np.mean(delta, axis=1, keepdims=True)

    def reset_grads(self):
        for w in self._W_grads:
            w.fill(0)
        for b in self._b_grads:
            b.fill(0)

    def _reset_cache(self):
        self._As = []
        self._Zs = []


class AdamOptimizer:
    def __init__(self, beta1, beta2, epsilon, learning_rate):
        self._beta1 = beta1
        self._beta2 = beta2
        self._learning_rate = learning_rate
        self._epsilon = epsilon
        self.m_weights = None
        self.v_weights = None
        self.m_biases = None
        self.v_biases = None
        self.iterations = 1

    def initialize(self, network: DeepNetwork):
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []
        for (w, b) in zip(network.weights, network.biases):
            self.m_weights.append(np.zeros_like(w))
            self.v_weights.append(np.zeros_like(w))
            self.m_biases.append(np.zeros_like(b))
            self.v_biases.append(np.zeros_like(b))

    def train(self, network: DeepNetwork, X_batch, Y_batch):
        network._feedforward_train(X_batch)
        network._backpropagate(Y_batch)
        self.update_moments_and_weights(network)

    def update_moments_and_weights(self, network: DeepNetwork):
        for i in range(len(network.weights)):
            self.m_weights[i] = self._beta1 * self.m_weights[i] + (1 - self._beta1) * network._W_grads[i]
            self.m_biases[i] = self._beta1 * self.m_biases[i] + (1 - self._beta1) * network._b_grads[i]
            self.v_weights[i] = self._beta2 * self.v_weights[i] + (1 - self._beta2) * np.square(network._W_grads[i])
            self.v_biases[i] = self._beta2 * self.v_biases[i] + (1 - self._beta2) * np.square(network._b_grads[i])

            m_w_hat = self.m_weights[i] / (1 - (self._beta1 ** self.iterations))
            m_b_hat = self.m_biases[i] / (1 - (self._beta1 ** self.iterations))
            v_w_hat = self.v_weights[i] / (1 - (self._beta2 ** self.iterations))
            v_b_hat = self.v_biases[i] / (1 - (self._beta2 ** self.iterations))

            network.weights[i] += (self._learning_rate / (np.sqrt(v_w_hat) + self._epsilon) * m_w_hat)
            network.biases[i] += (self._learning_rate / (np.sqrt(v_b_hat) + self._epsilon) * m_b_hat)
        self.iterations += 1

    def name(self):
        return 'Adam'


class SGDOptimizer:
    def __init__(self, learning_rate):
        self._learning_rate = learning_rate

    def initialize(self, network: DeepNetwork):
        pass

    def train(self, network: DeepNetwork, X_batch, Y_batch):
        network._feedforward_train(X_batch)
        network._backpropagate(Y_batch)
        self.update_weights(network)

    def update_weights(self, network: DeepNetwork):
        for i in range(len(network.weights)):
            network.weights[i] += self._learning_rate * network._W_grads[i]
            network.biases[i] += self._learning_rate * network._b_grads[i]

    def name(self):
        return 'SGD'


class MomentumOptimizer:
    def __init__(self, gamma, learning_rate):
        self._gamma = gamma
        self._learning_rate = learning_rate
        self._weights_momentum = None
        self._biases_momentum = None

    def initialize(self, network: DeepNetwork):
        self._weights_momentum = []
        self._biases_momentum = []
        for (w, b) in zip(network.weights, network.biases):
            self._weights_momentum.append(np.zeros_like(w))
            self._biases_momentum.append(np.zeros_like(b))

    def train(self, network: DeepNetwork, X_batch, Y_batch):
        network._feedforward_train(X_batch)
        network._backpropagate(Y_batch)
        self.update_momentum_and_weights(network)

    def update_momentum_and_weights(self, network: DeepNetwork):
        for i in range(len(network.weights)):
            self._weights_momentum[i] = self._gamma * self._weights_momentum[i] + self._learning_rate * network._W_grads[i]
            self._biases_momentum[i] = self._gamma * self._biases_momentum[i] + self._learning_rate * network._b_grads[i]
            network.weights[i] += self._weights_momentum[i]
            network.biases[i] += self._biases_momentum[i]

    def name(self):
        return 'Momentum'


class AdaGradOptimizer:
    def __init__(self, epsilon, learning_rate):
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._w_grads_square_sum = None
        self._b_grad_square_sum = None

    def initialize(self, network: DeepNetwork):
        self._w_grads_square_sum = []
        self._b_grad_square_sum = []
        for (w, b) in zip(network.weights, network.biases):
            self._w_grads_square_sum.append(np.zeros_like(w))
            self._b_grad_square_sum.append(np.zeros_like(b))

    def train(self, network: DeepNetwork, X_batch, Y_batch):
        network._feedforward_train(X_batch)
        network._backpropagate(Y_batch)
        self.update_sums_and_weights(network)

    def update_sums_and_weights(self, network: DeepNetwork):
        for i in range(len(network.weights)):
            self._w_grads_square_sum[i] += np.square(network._W_grads[i])
            self._b_grad_square_sum[i] += np.square(network._b_grads[i])
            network.weights[i] += (self._learning_rate / np.sqrt(self._w_grads_square_sum[i] + self._epsilon)) * network._W_grads[i]
            network.biases[i] += (self._learning_rate / np.sqrt(self._b_grad_square_sum[i] + self._epsilon)) * network._b_grads[i]

    def name(self):
        return 'AdaGrad'


class AdaDeltaOptimizer:
    def __init__(self, gamma, epsilon):
        self._gamma = gamma
        self._epsilon = epsilon
        self._w_grad_square_sum = None
        self._b_grad_square_sum = None
        self._w_delta_square_sum = None
        self._b_delta_square_sum = None

    def initialize(self, network: DeepNetwork):
        self._w_grad_square_sum = []
        self._b_grad_square_sum = []
        self._w_delta_square_sum = []
        self._b_delta_square_sum = []
        for (w, b) in zip(network.weights, network.biases):
            self._w_grad_square_sum.append(np.zeros_like(w))
            self._b_grad_square_sum.append(np.zeros_like(b))
            self._w_delta_square_sum.append(np.zeros_like(w))
            self._b_delta_square_sum.append(np.zeros_like(b))

    def train(self, network: DeepNetwork, X_batch, Y_batch):
        network._feedforward_train(X_batch)
        network._backpropagate(Y_batch)
        self.update_sums_and_weights(network)

    def update_sums_and_weights(self, network: DeepNetwork):
        for i in range(len(network.weights)):
            self._w_grad_square_sum[i] = self._gamma * self._w_grad_square_sum[i] + (1 - self._gamma) * np.square(network._W_grads[i])
            self._b_grad_square_sum[i] = self._gamma * self._b_grad_square_sum[i] + (1 - self._gamma) * np.square(network._b_grads[i])

            delta_w = np.sqrt(self._w_delta_square_sum[i] + self._epsilon) / np.sqrt(self._w_grad_square_sum[i] + self._epsilon) * network._W_grads[i]
            delta_b = np.sqrt(self._b_delta_square_sum[i] + self._epsilon) / np.sqrt(self._b_grad_square_sum[i] + self._epsilon) * network._b_grads[i]

            network.weights[i] += delta_w
            network.biases[i] += delta_b

            self._w_delta_square_sum[i] = (self._gamma * self._w_delta_square_sum[i] + (1 - self._gamma) * np.square(delta_w))
            self._b_grad_square_sum[i] = (self._gamma * self._b_delta_square_sum[i] + (1 - self._gamma) * np.square(delta_b))

    def name(self):
        return 'AdaDelta'

def read(name):
    with open(name, 'rb') as data:
        datasets = pkl.load(data, encoding='latin1')
        [X_train, Y_train], [X_val, Y_val], [X_test, Y_test] = datasets
        X_train = X_train.T
        Y_train = np.reshape(Y_train, (np.shape(Y_train)[0], 1))
        X_val = X_val.T
        Y_val = np.reshape(Y_val, (np.shape(Y_val)[0], 1))
        X_test = X_test.T
        Y_test = np.reshape(Y_test, (np.shape(Y_test)[0], 1))
        return X_train, Y_train, X_val, Y_val, X_test,  Y_test


def load_dataset():
    K_CLASSES = 10
    X_train, Y_train, X_val, Y_val, X_test, Y_test = read('mnist.pkl')
    Y_train_one, Y_val_one, Y_test_one = one_hot(Y_train, K_CLASSES), one_hot(Y_val, K_CLASSES), one_hot(Y_test,
                                                                                                         K_CLASSES)
    return X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, X_test, Y_test, Y_test_one


def save_list_csv(filename, lst, header=None):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerow(lst)

def run_optimizer_tests():
    K_CLASSES = 10
    LEARNING_RATE = 0.001
    EPOCHS = 50
    BATCH_SIZE = 50
    N_HIDDEN = 50

    adamOptimizer = AdamOptimizer(0.9, 0.999, 10e-8, LEARNING_RATE)
    sgdOptimizer = SGDOptimizer(LEARNING_RATE)
    momentumOptimizer = MomentumOptimizer(0.9, LEARNING_RATE)
    adaGradOptimizer = AdaGradOptimizer(10e-8, LEARNING_RATE)
    adaDeltaOptimizer = AdaDeltaOptimizer(0.9, 10e-8)

    normalInitializer = NormalInitializer(0, 0.05)

    X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, X_test, Y_test, Y_test_one = load_dataset()
    optimizers = [sgdOptimizer, momentumOptimizer, adaGradOptimizer, adaDeltaOptimizer, adamOptimizer]

    train_losses, val_losses, val_errors = [], [], []
    for optimizer in optimizers:
        test = DeepNetwork([np.shape(X_train)[0], N_HIDDEN, K_CLASSES], ActivationFunction(relu, relu_derivative),
                           ActivationFunction(sigmoid, sigmoid_derivative), normalInitializer)
        train_loss, val_loss, val_error = test.train(X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, EPOCHS,
                                                     BATCH_SIZE, optimizer)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_errors.append(val_error)

    fig = plt.figure()
    epochs = np.arange(0, EPOCHS)
    for i, loss in enumerate(train_losses):
        plt.plot(epochs, loss, label=optimizers[i].name())

    plt.xlabel('Numer epoki')
    plt.ylabel('Wartość funkcji kosztu na ciągu treningowym')
    plt.title('Koszt na ciągu treningowym')
    plt.legend()
    plt.show()
    fig.savefig('opt_train_losses.png', dpi=fig.dpi)

    fig = plt.figure()
    epochs = np.arange(0, EPOCHS)
    for i, loss in enumerate(val_losses):
        plt.plot(epochs, loss, label=optimizers[i].name())

    plt.xlabel('Numer epoki')
    plt.ylabel('Wartość funkcji kosztu na ciągu walidacyjnym')
    plt.title('Koszt na ciągu walidacyjnym')
    plt.legend()
    plt.show()
    fig.savefig('opt_val_losses.png', dpi=fig.dpi)

    fig = plt.figure()
    epochs = np.arange(0, EPOCHS)
    for i, err in enumerate([[error * 100 for error in errors] for errors in val_errors]):
        plt.plot(epochs, err, label=optimizers[i].name())

    plt.xlabel('Numer epoki')
    plt.ylabel('Błąd klasyfikacji[%]')
    plt.title('Błąd klasyfikacji na ciągu walidacyjnym')
    plt.legend()
    plt.show()
    fig.savefig('opt_val_errors.png', dpi=fig.dpi)

def run_initializer_tests():
    K_CLASSES = 10
    LEARNING_RATE = 0.01
    EPOCHS = 50
    BATCH_SIZE = 50
    N_HIDDEN = 50
    normalInitializer = NormalInitializer(0, 0.05)
    xavierInitializer = XavierInitializer()
    heInitializer = HeInitializer(0, 0.05)
    initializers = [normalInitializer, xavierInitializer, heInitializer]

    # optimizer = AdamOptimizer(0.9, 0.999, 10e-8, LEARNING_RATE)
    optimizer = SGDOptimizer(LEARNING_RATE)

    X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, X_test, Y_test, Y_test_one = load_dataset()
    train_losses, val_losses, val_errors = [], [], []
    for initializer in initializers:
        test = DeepNetwork([np.shape(X_train)[0], N_HIDDEN, K_CLASSES], ActivationFunction(relu, relu_derivative),
                           ActivationFunction(sigmoid, sigmoid_derivative), initializer)
        train_loss, val_loss, val_error = test.train(X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, EPOCHS,
                                                     BATCH_SIZE, optimizer)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_errors.append(val_error)

    fig = plt.figure()
    epochs = np.arange(0, EPOCHS)
    for i, loss in enumerate(train_losses):
        plt.plot(epochs, loss, label=initializers[i].name())

    plt.xlabel('Numer epoki')
    plt.ylabel('Wartość funkcji kosztu na ciągu treningowym')
    plt.title('Koszt na ciągu treningowym')
    plt.legend()
    plt.show()
    fig.savefig('init_train_losses.png', dpi=fig.dpi)

    fig = plt.figure()
    epochs = np.arange(0, EPOCHS)
    for i, loss in enumerate(val_losses):
        plt.plot(epochs, loss, label=initializers[i].name())

    plt.xlabel('Numer epoki')
    plt.ylabel('Wartość funkcji kosztu na ciągu walidacyjnym')
    plt.title('Koszt na ciągu walidacyjnym')
    plt.legend()
    plt.show()
    fig.savefig('init_val_losses.png', dpi=fig.dpi)

    fig = plt.figure()
    epochs = np.arange(0, EPOCHS)
    for i, err in enumerate([[error * 100 for error in errors] for errors in val_errors]):
        plt.plot(epochs, err, label=initializers[i].name())

    plt.xlabel('Numer epoki')
    plt.ylabel('Błąd klasyfikacji[%]')
    plt.title('Błąd klasyfikacji na ciągu walidacyjnym')
    plt.legend()
    plt.show()
    fig.savefig('init_val_errors.png', dpi=fig.dpi)

if __name__ == '__main__':
    K_CLASSES = 10
    LEARNING_RATE = 0.001
    EPOCHS = 50
    BATCH_SIZE = 50
    N_HIDDEN = 400
    initializer = XavierInitializer()
    optimizer = AdamOptimizer(0.9, 0.999, 10e-8, LEARNING_RATE)
    X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, X_test, Y_test, Y_test_one = load_dataset()
    test = DeepNetwork([np.shape(X_train)[0], N_HIDDEN, K_CLASSES], ActivationFunction(relu, relu_derivative),
                       ActivationFunction(sigmoid, sigmoid_derivative), initializer)
    train_loss, val_loss, val_error = test.train(X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, EPOCHS,
                                                 BATCH_SIZE, optimizer)