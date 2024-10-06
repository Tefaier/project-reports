import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import random
from tqdm import tqdm

def targetFunction(x): # 3 -> 2
    return torch.tensor([10*x[0] - 8*x[2], x[0] + 5*x[1] + 2*x[2]], dtype=torch.float32)

def getDataSample():
    inp = torch.rand(3) * 10
    target = targetFunction(inp)
    return inp, target

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fcl = nn.Linear(3, 2)

    def forward(self, x):
        x = self.fcl(x)
        return x

# non linear type, due to tanh, but nn. should be used rather
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(1, 5)
#         self.fc2 = nn.Linear(1, 5)
#
#     def forward(self, x):
#         x = torch.tanh(self.fc1(x))
#         x = self.fc2(x)
#         return x

# for classification in 3D
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(3, 10)
#         self.fc2 = nn.Linear(10, 5)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return torch.sigmoid(x)

# for each task appropriate set of layers, activation functions, learning rate should be chosen

net = Net()
print("Model parameters")
for param in net.parameters():
    print(param)

# use BCELoss for binary type classification
# optim.Adam can change learning rate during work
criterion = nn.MSELoss()
learningRate = 0.01
optimizer = optim.SGD(net.parameters(), lr=learningRate, momentum=0.9)

epochs = 300
outputPeriod = 100
for i in range(0, epochs):
    inp, target = getDataSample()
    optimizer.zero_grad()
    outputs = net(inp)
    loss = criterion(outputs, target)
    lossValue = loss.item()
    if (i % outputPeriod == 0):
        print(f"Epoch {i}. Model parameters")
        for param in net.parameters():
            print(param)
        print(f"Current loss: {lossValue}")
    loss.backward()
    optimizer.step()

print(f"Epoch {epochs}. Model parameters")
for param in net.parameters():
    print(param)

for i in range(0, 5):
    inp, target = getDataSample()
    outputs = net(inp)
    loss = criterion(outputs, target)
    print(f"input {inp} output {outputs} target {target}\nloss {loss}\n")

for i in range(0, 5):
    inp =torch.rand(3) * 100
    target = targetFunction(inp)
    outputs = net(inp)
    loss = criterion(outputs, target)
    print(f"input {inp} output {outputs} target {target}\nloss {loss}\n")

# tqdm is used to show cool console things like progress bar