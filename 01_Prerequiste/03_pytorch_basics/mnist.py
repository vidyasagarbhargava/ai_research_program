import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from termcolor import colored, cprint
#0) hyperparameters
device = torch.device("mps" if torch.backends.mps.is_available() else 'cpu')
input_size = 784
hidden_size = 500
num_classes = 10
batch_size = 32

#1) Dataset creation
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#2) Design Model
class NeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size,hidden_size,num_classes).to(device)

#3) Define Loss and Optimizer
num_epochs = 10
learning_rate = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#4) Training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images) #forward pass
        loss = criterion(outputs, labels) #loss
        loss.backward() #calculate gradient
        optimizer.step() #update weights
        optimizer.zero_grad() #zero gradient after updating
        if(i+1) %100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item(): .4f}')

#5) Test the model
with torch.no_grad():
    n_correct = 0 
    n_samples = len(test_loader.dataset)

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()


    acc = n_correct/n_samples
    #cprint(f'Accuracy  of the network on the {n_samples} test images:{100*acc} %')
    print(colored(f'Accuracy  of the network on the {n_samples} test images:{100*acc} %','red'))

