import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ftrl import FTRL

# Hyperparameters
batch_size = 100

input_size = 784
output_size = 10

ftrl_alpha = 1.0
ftrl_beta = 1.0
ftrl_l1 = 1.0
ftrl_l2 = 1.0


# Dataset
traindata = datasets.MNIST(
    "./data", train=True, transform=transforms.ToTensor(), download=True
)
testdata = datasets.MNIST(
    "./data", train=False, transform=transforms.ToTensor(), download=True
)
trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False)


# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.W = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.W(x)


model = LogisticRegression(input_size, output_size)
loss_fn = nn.CrossEntropyLoss()

optimizer = FTRL(
    model.parameters(), alpha=ftrl_alpha, beta=ftrl_beta, l1=ftrl_l1, l2=ftrl_l2
)

# Train
iter = 0
for images, labels in trainloader:
    images = images.view(-1, input_size)
    optimizer.zero_grad()
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    iter += 1
    if iter % 100 == 0:
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.view(-1, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        accuracy = 100 * correct / total
        print(
            "Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy)
        )
