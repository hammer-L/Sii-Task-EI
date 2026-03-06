"""
network model utils
"""

import torch
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

my_test = torch.tensor([[0., 0., 0.],
                        [-1., -1., -1.],
                        [1., 1., 1.]])

# print (my_test)

def relu(x):
    return torch.maximum(x, torch.zeros_like(x))
# print (relu(my_test))

def relu_derivative(x):
    return (x > 0).float()

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))
# print (sigmoid(my_test))

def sigmoid_derivative(s):
    return s * (1 - s)

def mse_loss(y_pre, y):
    return torch.mean((y_pre - y) ** 2)

def softmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
    return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

def crossentropy(y_pre, y):
    y = y.long()
    log_probs = torch.log(y_pre.gather(1, y.unsqueeze(1)).squeeze(1) + 1e-9)
    return -torch.mean(log_probs)

"""
2. model class
 - 4 layer
 - 5 classifications
 - softmax
 - CrossEntropy loss
"""
class Model():
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        self.W1 = torch.randn(input_size, hidden_size).to(device) * math.sqrt(2.0 / input_size)  # Xavier初始化
        self.W2 = torch.randn(hidden_size, 2 * hidden_size).to(device) * math.sqrt(2.0 / hidden_size)
        self.W3 = torch.randn(2 * hidden_size, hidden_size).to(device) * math.sqrt(2.0 / (2 * hidden_size))
        self.W4 = torch.randn(hidden_size, output_size).to(device) * math.sqrt(2.0 / hidden_size)

        self.b1 = torch.zeros(1, hidden_size).to(device)
        self.b2 = torch.zeros(1, 2 * hidden_size).to(device)
        self.b3 = torch.zeros(1, hidden_size).to(device)
        self.b4 = torch.zeros(1, output_size).to(device)

        self.cache = {}

    def forward(self, x):
        self.z1 = torch.matmul(x, self.W1) + self.b1
        self.A1 = relu(self.z1)
        self.z2 = torch.matmul(self.A1, self.W2) + self.b2
        self.A2 = relu(self.z2)
        self.z3 = torch.matmul(self.A2, self.W3) + self.b3
        self.A3 = relu(self.z3)
        self.z4 = torch.matmul(self.A3, self.W4) + self.b4
        self.y_pre = softmax(self.z4)

        self.cache['x'] = x
        self.cache['A1'] = self.A1
        self.cache['A2'] = self.A2
        self.cache['A3'] = self.A3
        self.cache['z4'] = self.z4

        return self.y_pre

    def backward(self, y_pre, y):
        """
        y_pre: [Batch_Size, Num_Classes]
        y:     [Batch_Size]
        """
        batch_size = y_pre.shape[0]
        x = self.cache['x']
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        A3 = self.cache['A3']

        # 改为 one-hot编码
        if y.dim() == 1:
            # 创建全零矩阵 [Batch, Classes]
            y_one_hot = torch.zeros_like(y_pre)
            # 使用 scatter_ 将 1 填入对应位置
            y_one_hot.scatter_(1, y.unsqueeze(1), 1.0)
        else:
            y_one_hot = y

        dz4 = (y_pre - y_one_hot) / batch_size

        # layer 4
        self.dW4 = torch.matmul(A3.T, dz4)
        self.db4 = torch.sum(dz4, dim=0, keepdim=True)

        #layer 3
        da3 = torch.matmul(dz4, self.W4.t())
        dz3 = da3 * (A3 > 0).float()
        self.dW3 = torch.matmul(A2.T, dz3)
        self.db3 = torch.sum(dz3, dim=0, keepdim=True)

        # layer 2
        da2 = torch.matmul(dz3, self.W3.t())
        dz2 = da2 * (A2 > 0).float()
        self.dW2 = torch.matmul(A1.T, dz2)
        self.db2 = torch.sum(dz2, dim=0, keepdim=True)

        # layer 1
        da1 = torch.matmul(dz2, self.W2.t())
        dz1 = da1 * (A1 > 0).float()
        self.dW1 = torch.matmul(x.T, dz1)
        self.db1 = torch.sum(dz1, dim=0, keepdim=True)

        # 更新参数
        self.step()

    def step(self):
        with torch.no_grad():
            self.W1 -= self.lr * self.dW1
            self.b1 -= self.lr * self.db1
            self.W2 -= self.lr * self.dW2
            self.b2 -= self.lr * self.db2
            self.W3 -= self.lr * self.dW3
            self.b3 -= self.lr * self.db3
            self.W4 -= self.lr * self.dW4
            self.b4 -= self.lr * self.db4

def evaluate(model, x, y):
    with torch.no_grad():
        y_pre = model.forward(x)
        loss = crossentropy(y_pre, y)
        preds = torch.argmax(y_pre, dim=1)
        acc = torch.mean((preds == y).float()).item()
    return loss, acc






