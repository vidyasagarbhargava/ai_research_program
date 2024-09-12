import torch
import torch.nn as nn 

# create some data 
X = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8],[10],[12],[14],[16]], dtype=torch.float32)

n_samples,n_features = X.shape
X_test = torch.tensor([5],dtype = torch.float32)


#1) Design model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self,x):
        return self.lin(x)

input_size, output_size = n_features, n_features
model = LinearRegression(input_size, output_size)
print(f"Prediction before training : f({X_test.item()}) = {model(X_test).item():.3f}")

#2) Define loss and optimizer
learning_rate = 0.01
n_epochs = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

#3) Training loop
for epoch in range(n_epochs):
    y_predicted = model(X) #forward pass
    l = loss(Y, y_predicted) #loss
    l.backward() #calculate gradient
    optimizer.step() #update weights
    optimizer.zero_grad() #zero gradient after updating

    if (epoch+1)%10 == 0:
        w,b = model.parameters()
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l.item())

print(f"Prediction after training : f({X_test.item()}) = {model(X_test).item():.3f}")

