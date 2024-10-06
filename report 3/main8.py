from marshal import loads

import torch
import torchvision
import torchvision.transforms as tr
import torch.utils.data
from sympy.physics.units import momentum
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import models
from torchsummary import summary

from main3 import outputs, learningRate

batchSizeTrain = 64
batchSizeTest = 1024

imageTransform = tr.Compose([
    tr.ToTensor(),
    tr.Normalize((0.1307,), (0.3081,))
])

trainDataset = torchvision.datasets.MNIST(
    'dataset/',
    train=True,
    download=True,
    transform=imageTransform
)

testDataset = torchvision.datasets.MNIST(
    'dataset/',
    train=False,
    download=True,
    transform=imageTransform
)

trainLoader = DataLoader(trainDataset, batch_size=batchSizeTrain, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=batchSizeTest, shuffle=True)
lossFunction = nn.CrossEntropyLoss()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(model: nn.Module, device: torch.device, trainLoader: DataLoader, optimizer: Optimizer, epoch: int, logInterval=100):
    model.train()
    tk0 = tqdm(trainLoader, total=int(len(trainLoader)))
    counter = 0
    trainLoss = 0
    for batchIDX, (data, target) in enumerate(tk0):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.view(-1, 28*28))
        loss = lossFunction(output, target)
        trainLoss += lossFunction(output, target).item()
        loss.backward()
        optimizer.step()
        counter += 1
        tk0.set_postfix(loss=(loss.item() * data.size(0) / (counter * trainLoader.batch_size)))
    trainLoss /= len(trainLoader.dataset)
    tk0.close()
    return trainLoss

def test(model: nn.Module, device: torch.device, testLoader: DataLoader):
    model.eval()
    testLoss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(-1, 28*28))
            loss = lossFunction(output, target)
            testLoss += lossFunction(output, target).item()
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print("Accuracy of: " + str(100.0 * correct / len(testLoader.dataset)))
    testLoss /= len(testLoader.dataset)
    return testLoss

learningRate = 0.01
momentum = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=learningRate)

numEpochs = 3
trainLossEpochs = []
testLossEpochs = []
for epoch in range(1, numEpochs + 1):
    trainLossEpochs.append(train(model, device, trainLoader, optimizer, epoch))
    testLossEpochs.append(test(model, device, testLoader))
plt.plot(trainLossEpochs)
plt.plot(trainLossEpochs)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

print(summary(model, (1, 28 * 28)))
# 95 accuracy, 80 it/s, 50k parameters
