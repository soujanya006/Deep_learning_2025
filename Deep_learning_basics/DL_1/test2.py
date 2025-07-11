import numpy as np

import pandas as pd

import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

print("Hello___Testing : ")


if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available. Using CPU.")

df = pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')
#print(df.head())
print(df.shape)


#Pre processing

df.drop(columns=['id', 'Unnamed: 32'], inplace= True)

#train test split
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.2)


#scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Label Encoding
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)


#Numpy arrays to PyTorch tensors
X_train_tensor= torch.from_numpy(X_train)
X_test_tensor=torch.from_numpy(X_test)

y_train_tensor=torch.from_numpy(y_train)
y_test_tensor=torch.from_numpy(y_test)

print(X_train_tensor[2])

print(X_train_tensor.shape,y_train_tensor.shape)


#defining the model

class My_Simple_NN():

    def __init__(self,X):
        self.weights=torch.rand(X.shape[1],1,dtype=torch.float64,requires_grad=True)
        self.bias=torch.zeros(1,dtype=torch.float64,requires_grad=True)

    def forward(self, X):
        z = torch.matmul(X, self.weights) + self.bias
        y_pred = torch.sigmoid(z)
        return y_pred
        
    def loss_function(self, y_pred, y):

        # Clamp predictions to avoid log(0)
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)

        # Calculate loss
        loss = -(y_train_tensor * torch.log(y_pred) + (1 - y_train_tensor) * torch.log(1 - y_pred)).mean()
        return loss 
        
#important Parameters
learning_rate= 0.1
epochs= 30


# create model
model = My_Simple_NN(X_train_tensor)

# define loop
for epoch in range(epochs):

  # forward pass
  y_pred = model.forward(X_train_tensor)

  # loss calculate
  loss = model.loss_function(y_pred, y_train_tensor)

  # backward pass
  loss.backward()
  

  # parameters update
  with torch.no_grad():
    model.weights -= learning_rate * model.weights.grad
    model.bias -= learning_rate * model.bias.grad

  # zero gradients
  model.weights.grad.zero_()
  model.bias.grad.zero_()

  # print loss in each epoch
print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

