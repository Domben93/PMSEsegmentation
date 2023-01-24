import numpy as np
import csv
from sklearn.metrics import confusion_matrix
class MLP:

    def __init__(self, input_size, layers):

        self.w1 = np.random.uniform(-0.01, 0.01, size=(layers, input_size))
        self.w2 = np.random.uniform(-0.01, 0.01, size=(1, self.w1.shape[0]))
        self.bias1 = np.ones((layers, 1))
        self.bias2 = np.ones((1, 1))
        self.hidden_activation = self.ReLU
        self.last_activation = self.sigmoid
        self.hidden_activation_deriv = self.ReLU_derivative
        self.last_activation_deriv = self.sigmoid_derivative

        self.loss = None

        # backwards variables
        self.input = 0
        self.a1 = 0
        self.a2 = 0
        self.z1 = 0
        self.z2 = 0

    def forward(self, x):

        self.input = x

        self.z1 = np.dot(self.w1, self.input.T) + self.bias1

        self.a1 = self.hidden_activation(self.z1)

        self.z2 = np.dot(self.w2, self.a1) + self.bias2

        return self.last_activation(self.z2)

    def backward(self, deriv_loss, lr):

        dz = self.last_activation_deriv(self.z2) * np.sum(deriv_loss)
        dw = np.dot(dz, self.a1.T)

        db = np.sum(dz, axis=1, keepdims=True) / 1207

        self.w2 = self.w2 - lr * dw
        self.bias2 = self.bias2 - lr * db
        print(self.bias2)

        dz = dz * np.dot(self.w2, self.hidden_activation_deriv(self.z1))
        dw = np.dot(dz, self.input)
        db = np.sum(dz, axis=1, keepdims=True) / 1207

        self.w1 = self.w1 - lr * dw

        self.bias1 = self.bias1 - lr * db
        print(self.bias1)

    def fit(self, x, label, it=100, lr=0.01, disp_loss=True):

        for i in range(it):

            pred = self.forward(x)

            self.loss = self.binary_cross_entropy(pred, label)

            back_loss = self.binary_deriv(pred, label)

            self.backward(back_loss, lr)

            #print(f'iter: {i}')
            #print(np.mean(self.loss))



    @staticmethod
    def ReLU(x):
        return np.where(x >= 2, x, 0)

    @staticmethod
    def ReLU_derivative(x):
        return np.where(x >= 0, 1, 0)

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))

    @staticmethod
    def cost_func(pred, label):
        return (label - pred)**2

    @staticmethod
    def loss_derivative(pred, label):
        return -2 * (label * pred)

    @staticmethod
    def binary_cross_entropy(pred, label):
        eps = 0.0001
        return -(label * np.log(pred + eps) + (1 - label)*np.log(1 - pred + eps))

    @staticmethod
    def binary_deriv(pred, label):
        return (pred - label) / ((pred * (1 - pred)) + 0.0001)


if __name__ == '__main__':

    train_data = np.loadtxt('seals_train.csv', delimiter=' ')
    test_data = np.loadtxt('seals_test.csv', delimiter=' ')

    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    x_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    print(x_train.shape)
    mlp = MLP(128, 10)

    mlp.fit(x_train, y_train, it=2, lr=0.01)

    #print(mlp.bias1.shape)
    #print(mlp.bias2.shape)
    #pred = mlp.forward(x_test)

    #pred = np.where(pred >= 0.5, 1, 0)

    #cm = confusion_matrix(pred, y_test)
    #print(cm)
    #x = np.array([[10, 10], [3, 4], [6, 4], [4, 3], [3, 5], [-1, -1], [-10, -4], [-5, -3], [-5, -9], [-4, -6]])
    #label = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    #mlp.fit(x, label,it=200, lr=0.1)

    #print(mlp.w1)
    #print(mlp.w2)