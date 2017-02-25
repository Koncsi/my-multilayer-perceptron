import numpy as np

# Simple MLP
class MLP:
    # parameter_array is an array that contains the architecture of the network
    # parameter_array = [2,3,4,1] -> input dimension=2, two hidden layers with 3 ans 4 neurons, output layer with dimension=1
    def __init__(self, parameter_array, act_function='sigmoid'):
        # storing the depth for iterations
        self.depth = len(parameter_array)
        # setting the activation function
        self.activation_unit = act_function
        # only for inspection
        self.layers_struc = {'input'    :parameter_array[0],
                             'hidden'   :parameter_array[1:-1],
                             'output'   :parameter_array[-1]}

        # creating neurons
        self.layers = []
        for i in range(self.depth):
            self.layers.append(np.ones(parameter_array[i]))

        # creating weights, inited with zeros
        self.weights = []
        for i in range(self.depth-1):
            self.weights.append(np.zeros([len(self.layers[i]),len(self.layers[i+1])]))

        # setting different values than zero for better learning
        self.init_weights()

    # propagate the input value forward
    def forward(self, x):
        self.layers[0] = x
        for i in range(1,self.depth):
            self.layers[i] = self.activation(np.dot(self.layers[i-1], self.weights[i-1]))

        return self.layers[-1]

    # with the given input computes the error and the gradients
    def backward(self, true_y, x):
        current_y = self.forward(x)
        error = true_y-current_y
        # computing deltas
        delta = -(error)*self.d_activation(np.dot(self.layers[-2], self.weights[-1]))
        deltas = []
        deltas.append(delta)

        for i in range(self.depth-2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.d_activation(np.dot(self.layers[i-1], self.weights[i-1]))
            deltas.insert(0, delta)

        # computing gradients for each weights
        delta_w = []
        for i in range(len(deltas)):
            # matrix from vector
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            delta_w.append(np.dot(layer.T, delta))

        # returns the error and the gradients
        return np.power(error,2).sum(), delta_w

    # the learning algorithm
    def learn(self,samples, epochs=5000, lrate=0.1, opt_method='SGD'):
        for i in range(epochs):
            n = np.random.randint(samples.size)
            error, d_W = self.backward(samples['y'][n], samples['x'][n])

            # updating weights
            for j in range(len(self.weights)):
                self.weights[j] += -lrate*d_W[j]

            # print the current error
            if i%10000 == 0:
                print(i, '. epoch, error: ', error)

    # sigmoid activation function
    # TODO: create othe activation functions!
    def activation(self, x):
        return {
            'sigmoid': sigmoid(x),
            'relu'   : relu(x),
            'lrelu'  : leaky_relu(x)
        }[self.activation_unit]

    # derivative of the sigmoid function
    def d_activation(self, x):
        return {
            'sigmoid': d_sigmoid(x),
            'relu': d_relu(x),
            'lrelu': d_leaky_relu(x)
        }[self.activation_unit]

    # init weights with random values, between 0,1
    def init_weights(self):
        for i in range(self.depth-1):
            self.weights[i] = np.random.uniform(-1.0, 1.0, size=(len(self.layers[i]),len(self.layers[i+1])))

    # for debugging
    def print_mlp_structure(self):
        print('input dimension: '   + str(self.layers_struc['input']))
        print('hidden dimensions: ' + str(self.layers_struc['hidden']))
        print('output dimension: '  + str(self.layers_struc['output']))

    # for debugging
    def print_weights(self):
        for i in range(self.depth-1):
            print(self.weights[i])

# sigmoid activation function and it's derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def d_sigmoid(x):
    return np.exp(-x) / np.power(1 + np.exp(-x), 2)

# rectified linear unit activation func and it's derivative
def relu(x):
    return np.maximum(x,0)
def d_relu(x):
    return 1. * (x > 0)

# leaky relu, and it's derivative
def leaky_relu(x):
    return np.maximum(x,0.01*x)
def d_leaky_relu(x):
    return 1. * (x > 0) + 0.1 * (x < 0)
