import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print(torch.tensor(np.array([[1, -1], [-1, 1]], dtype=float)))
print(torch.zeros([2, 4]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.ones([2, 4], dtype=torch.float64, device=device))

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[1, -1], [-1, 4]])

print(x.sum().item())
# performs operation between corresponding cells one by one
print(torch.add(x, y))
print(torch.mul(x, y))

weights = torch.rand(3)
inp = torch.tensor([1, -1, 1], dtype=torch.float).t()
# matmul - matrix multiplication
print(torch.matmul(weights, inp).item())

# cat - concatenate, 0 - dimension of concatenation
print(torch.cat([weights, torch.tensor([10, 15])], 0))

# base network class provided by torch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x) + 1