import numpy as np

class ANN2:
    def __init__(self, population, input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.population = population
        self.size = (self.input_size * self.hidden_size) + self.hidden_size + (self.hidden_size * self.hidden_size2) + self.hidden_size2 + (self.hidden_size2 * self.output_size) + self.output_size 
        self.weight = np.random.normal(loc=0, scale=1, size=(self.population, self.size))
        self.weight1_size = (self.input_size * self.hidden_size)
        self.bias1_size = self.hidden_size
        self.hidden1_size = self.weight1_size + self.bias1_size
        self.weight2_size = (self.hidden_size * self.hidden_size)
        self.bias2_size = (self.hidden_size * self.output_size)
        self.hidden2_size = self.hidden1_size + self.weight2_size + self.bias2_size
        self.weight3_size = (self.hidden_size * self.output_size)
        self.bias3_size = self.output_size

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def reLU(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        self.hidden1 = np.array([np.dot(x[i], self.weight[i, :self.weight1_size].reshape(self.input_size, self.hidden_size))\
                                  + self.weight[i, self.weight1_size:self.weight1_size + self.hidden_size] for i in range(self.population)])
        self.hidden2 = np.array([np.dot(self.hidden1[i], self.weight[i, self.hidden1_size:self.hidden1_size + self.weight2_size].reshape(self.hidden1.shape[1], self.hidden_size))\
                                  + self.weight[i, self.hidden1_size + self.weight2_size: self.hidden1_size + self.weight2_size + self.bias2_size] for i in range(self.population)])
        self.output = np.array([self.sigmoid(np.dot(self.hidden2[i], self.weight[i, self.hidden2_size: self.hidden2_size + self.weight3_size])\
                                              + self.weight[i, self.hidden2_size + self.weight3_size: self.hidden2_size + self.weight3_size + self.bias3_size]) for i in range(self.population)])
        return np.where(self.output > 0.5, 1, 0)
    
