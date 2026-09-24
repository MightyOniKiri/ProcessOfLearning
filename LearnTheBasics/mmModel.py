import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.faltten = nn.Flatten()
        
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.faltten(x)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        logits = self.fc3(x)
        
        return logits
    
model = NeuralNetwork()
print(model)