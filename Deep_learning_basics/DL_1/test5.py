import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, input_features):

        super().__init__() # ------> child class constructor , calling parent class constructor(nn module)

        self.linear1 = nn.Linear(input_features, 3) ## 5 input features are going ## we make the skeleton of the layers
        self.relu = nn.ReLU()
        self.linear2=nn.Linear(3,1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, features):
        out = self.linear1(features) # ---> the dataset goes here it take the [batch_size, input_features]
        out = self.relu(out)
        out= self.linear2(out)
        out=self.sigmoid(out)

        return out
    


# Define a permanent 10x5 tensor with fixed values (floats)
features = torch.tensor([
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.2, 0.6, 0.3, 0.9, 0.0],
    [0.7, 0.8, 0.4, 0.1, 0.3],
    [0.9, 0.2, 0.2, 0.2, 0.3],
    [0.3, 0.3, 0.6, 0.8, 0.1],
    [0.9, 0.7, 0.3, 0.5, 0.3],
    [0.1, 0.9, 0.8, 0.4, 0.4],
    [0.6, 0.5, 0.2, 0.7, 0.8],
    [0.3, 0.9, 0.9, 0.6, 0.2],
], dtype=torch.float32)

#print(features)

model= Model(features.shape[1]) # 5 input features are going 

# forword pass  model.forward(features) magic methods forward method gets triggered
model(features) 

print(model(features))





