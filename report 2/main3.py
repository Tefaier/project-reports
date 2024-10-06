import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import random
from tqdm import tqdm

def targetFunction(x):
    return 50 * x - 276

def getDataSample():
    inp = torch.tensor([np.random.random_sample()])
    target = targetFunction(inp)
    return inp, target

x = np.linspace(0, 1, 1000)
y = targetFunction(x) # math operation on np.array

lineRaw = go.Figure()
lineRaw.add_trace(go.Scatter(x=x, y=y, mode='lines', name='line'))
#lineRaw.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fcl = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fcl(x)
        return x

net = Net()
print("Model parameters")
for param in net.parameters():
    print(param)

criterion = nn.MSELoss()
learningRate = 0.1
optimizer = optim.SGD(net.parameters(), lr=learningRate)

epochs = 300
outputPeriod = 30
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
    print(f"input {inp} output {outputs.item()} target {target.item()}\nloss {loss}\n")

for i in range(0, 5):
    inp =torch.tensor([100 * np.random.random_sample()])
    target = targetFunction(inp)
    outputs = net(inp)
    loss = criterion(outputs, target)
    print(f"input {inp} output {outputs.item()} target {target.item()}\nloss {loss}\n")
