import numpy as np

class ANN:
    def __init__(self, population, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.population = population
        self.size = (self.input_size * self.hidden_size) + self.hidden_size + (self.hidden_size * self.output_size) + self.output_size
        self.weight = np.random.normal(loc=0, scale=0.1, size=(self.population, self.size))
        self.size1 = (self.input_size * self.hidden_size)
        self.size2 = (self.hidden_size * self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.hidden = np.array([np.dot(x[i], self.weight[i, :self.size1].reshape(self.input_size,self.hidden_size)) + self.weight[i, self.size1:self.size1 + self.hidden_size] for i in range(self.population)])
        self.output = np.array([self.sigmoid(np.dot(self.hidden[i], self.weight[i, self.size1 + self.hidden_size:self.size1 + self.hidden_size + self.size2]) + self.weight[i, self.size1 + self.hidden_size + self.size2:]) for i in range(self.population)])
        return np.where(self.output > 0.5, 1, 0)
    
