import numpy as np

# Simple MLP
class MLP:
    # parameter_array is an array that contains the architecture of the network
    # parameter_array = [2,3,4,1] -> input dimension=2, two hidden layers with 3 ans 4 neurons, output layer with dimension=1
    def __init__(self, parameter_array):
        # storing the depth for iterations
        self.depth = len(parameter_array)
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
    def learn(self,samples, epochs=5000, lrate=0.1):
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
        return 1/(1+np.exp(-x))

    # derivative of the sigmoid function
    def d_activation(self, x):
        return np.exp(-x)/np.power(1+np.exp(-x),2)

    # init weights with random values, between 0,1
    # TODO: normalize!
    def init_weights(self):
        for i in range(self.depth-1):
            self.weights[i] = np.random.rand(len(self.layers[i]),len(self.layers[i+1]))

    # for debugging
    def print_mlp_structure(self):
        print('input dimension: '   + str(self.layers_struc['input']))
        print('hidden dimensions: ' + str(self.layers_struc['hidden']))
        print('output dimension: '  + str(self.layers_struc['output']))

    # for debugging
    def print_weights(self):
        for i in range(self.depth-1):
            print(self.weights[i])