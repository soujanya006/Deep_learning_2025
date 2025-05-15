import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# 1) READ AND PREPROCESS THE DATA
url = 'https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv'
df = pd.read_csv(url)

# Inspect columns (uncomment to see columns)
# print(df.columns)

# The dataset usually has columns like: 'id', 'diagnosis', 'radius_mean', ..., 'Unnamed: 32'.
# Let's drop 'id' and 'Unnamed: 32' if they exist.
drop_cols = [col for col in ['id', 'Unnamed: 32'] if col in df.columns]
df.drop(columns=drop_cols, inplace=True)

# Encode diagnosis: M -> 1, B -> 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and labels
#  - 'diagnosis' is our target
#  - everything else is input features
X = df.drop('diagnosis', axis=1).values  # shape: [num_samples, num_features]
y = df['diagnosis'].values              # shape: [num_samples]

# Convert to PyTorch tensors (float for X, float for y)
X_tensor = torch.tensor(X, dtype=torch.float64)
y_tensor = torch.tensor(y, dtype=torch.float64)

# 2) DEFINE A SIMPLE MODEL
class SimpleBinaryClassifier(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # A small network: Linear -> ReLU -> Linear -> Sigmoid
        self.network = nn.Sequential(
            nn.Linear(num_features, 8),  # map from (num_features) to 8
            nn.ReLU(),
            nn.Linear(8, 1),            # map from 8 to 1
            nn.Sigmoid()                # squeeze output into [0,1]
        )
    
    def forward(self, x):
        return self.network(x)

# Instantiate the model
num_features = X_tensor.shape[1]  # number of columns in X
model = SimpleBinaryClassifier(num_features)

# 3) DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()                # binary cross-entropy lss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4) TRAIN THE MODEL
epochs = 20  # just a small number of epochs for demonstration

for epoch in range(epochs):
    # 1. Zero out any gradients from previous iterations
    optimizer.zero_grad()
    
    # 2. Forward pass: compute model output
    # shape of X_tensor: [num_samples, num_features]
    # shape of model output: [num_samples, 1]
    output = model(X_tensor)
    
    # 3. Compute the loss
    # 'output' is [num_samples, 1], so we use output.squeeze() to get [num_samples]
    loss = criterion(output.squeeze(), y_tensor)
    
    # 4. Backward pass: compute gradients
    loss.backward()
    
    # 5. Update parameters
    optimizer.step()
    
    # Print loss
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 5) FINAL OUTPUT
# After training, 'model' has learned parameters. We can use 'model(X_tensor)' 
# to get predictions on the entire dataset or any new sample.
