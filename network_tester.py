import MLP as nn
import numpy as np

# XOR for testing

network = nn.MLP([2,5,1])

samples = np.zeros(4, dtype=[('x', float, 2), ('y', float, 1)])
samples[0] = (0,0), 0
samples[1] = (1,0), 1
samples[2] = (0,1), 1
samples[3] = (1,1), 0

print('0,0 -> ' , str(network.forward([0,0])))
print('1,0 -> ' , str(network.forward([1,0])))
print('0,1 -> ' , str(network.forward([0,1])))
print('1,1 -> ' , str(network.forward([1,1])))

network.learn(samples,50000)

print('0,0 -> ' , str(network.forward([0,0])))
print('1,0 -> ' , str(network.forward([1,0])))
print('0,1 -> ' , str(network.forward([0,1])))
print('1,1 -> ' , str(network.forward([1,1])))