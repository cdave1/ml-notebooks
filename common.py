import torch
from torch import nn

# Define our model
# Inherits from nn.Module (so nn definitively stands for "neural network")
# We're creating the layers and structure of the neural network here.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # why 512? I get why 28*28 is the feature input.
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10) # 10 classes.
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits # is the name "logits" significant here? yes: https://deepai.org/machine-learning-glossary-and-terms/logit