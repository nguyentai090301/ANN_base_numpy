import numpy as np
class NeuralNetwork:
    def __init__(self, input_layer_size = 13, hidden_layer_size = 30, K = 2):
        self.weights1 = 0
        self.weights2 = 0
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.K = K
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def add_bias(self, X):
        m = len(X)
        bias = np.ones(m)
        X = np.vstack((bias, X.T)).T
        return X
    def initialize_random_weights(self,L_in,L_out):
        epsilon = 0.12
        W = np.zeros((L_in, L_out))
        W = np.random.rand(L_in,L_out)*(2*epsilon) - epsilon
        return W
    def deriv_sigmoid(self,output):
        return output * (1.0 - output)
    def remap(self, y):
        m = len(y)
        out = np.zeros((m, self.K))
        for index in range(m):
            out[index][y[index]] = 1
        return out

    def train(self, inputs, output, lr = 0.00002, epoch = 5000):
        x = self.add_bias(np.array(inputs))
        y = self.remap(np.array(output))
        self.weights1 = self.initialize_random_weights(self.input_layer_size, self.hidden_layer_size)
        self.weights2 = self.initialize_random_weights(self.hidden_layer_size, self.K)
        for i in range(epoch):
            print(f'Running on epoch {i}')
            l1 = self.sigmoid(np.dot(x,self.weights1))
            l2 = self.sigmoid(np.dot(l1,self.weights2))
            error = y - l2
            l2_del = error * self.deriv_sigmoid(l2)
            error0 = l2_del.dot(self.weights2.T)
            l1_del = error0 * self.deriv_sigmoid(l1)
            self.weights2 += lr*np.dot(l1.T,l2_del)
            self.weights1 += lr*np.dot(x.T,l1_del)
        return True
    def predict(self, inputs):
        x = self.add_bias(np.array(inputs))
        l1 = self.sigmoid(np.dot(x,self.weights1))
        l2 = self.sigmoid(np.dot(l1,self.weights2))
        y_pred = [np.argmax(l) for l in l2]
        return y_pred
    def compute_accuracy(self, y_true, y_preds):
        count_true = 0
        for index in range(len(y_true)):
            if y_true[index] == y_preds[index]:
                count_true += 1
        return round(count_true / len(y_true) * 100, 2)