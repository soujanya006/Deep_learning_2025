import torch
import torch.nn as nn

class SimpleBinaryClassifier(nn.Module):
    """
    A simple binary classification model using a single linear layer + sigmoid activation.
    
    Args:
        num_features (int): The number of input features for each sample.
    """
    def __init__(self, num_features):
        super().__init__()
        
        # 1) Define a linear layer that maps from 'num_features' to 1 output neuron
        self.linear_layer = nn.Linear(num_features, 1)
        
        # 2) Define a sigmoid activation function
        #    This will squash the linear layer's output into the range [0, 1].
        self.sigmoid_activation = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): The input data of shape [batch_size, num_features].
        
        Returns:
            Tensor: The output after applying the linear transformation and sigmoid,
                    typically of shape [batch_size, 1].
        """
        # 1) Apply the linear layer (y = W*x + b)
        linear_output = self.linear_layer(x)
        
        # 2) Apply the sigmoid activation (sigma(y))
        probability = self.sigmoid_activation(linear_output)
        
        return probability


# ------------------ Usage Example ------------------

if __name__ == "__main__":
    # Suppose you have 5 input features per sample
    num_features = 5
    
    # Instantiate the model
    model = SimpleBinaryClassifier(num_features)
    
    # Create a random input batch of 10 samples, each with 5 features
    sample_input = torch.rand(10, num_features)  # shape: [10, 5]
    
    # Perform the forward pass
    output = model(sample_input)
    
    # Output shape will be [10, 1] (one probability per sample)
    print("Output shape:", output.shape)
    print("Output values:\n", output)
