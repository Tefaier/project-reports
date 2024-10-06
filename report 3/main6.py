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

_, (datas, labels) = next(enumerate(testLoader))
sample = datas[0][0]
plt.imshow(sample, cmap='gray', interpolation='none')
plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train(model: nn.Module, device: torch.device, trainLoader: DataLoader, optimizer: Optimizer, epoch: int, logInterval=100):
    model.train()
    tk0 = tqdm(trainLoader, total=int(len(trainLoader)))
    counter = 0
    trainLoss = 0
    for batchIDX, (data, target) in enumerate(tk0):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        trainLoss += F.nll_loss(output, target, reduction='sum').item()
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
            output = model(data)
            loss = F.nll_loss(output, target)
            testLoss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print("Accuracy of: " + str(100.0 * correct / len(testLoader.dataset)))
    testLoss /= len(testLoader.dataset)
    return testLoss

learningRate = 0.01
momentum = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)

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

print(summary(model, (1, 28, 28)))

# convolution networks have much viewer nodes often, better work with colored images
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        # pay attention to how size changes
        self.conv = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.convDrop = nn.Dropout2d()
        self.fcl = nn.Linear(320, 50)

    def forward(self, x: torch.Tensor):
        # combination of using layers of itself and external as F.max_pool2d
        # to pass into usual layer x = x.view(-1, 320) to make it 3D->1D
        x.view()
        return F.log_softmax(x)

# cheatcode
alexnet = models.alexnet()
vgg = models.vgg19()