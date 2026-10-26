import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch

from torch.optim import lr_scheduler
from torchvision import models

from load_data import class_names, device
from train_model import train_model, visualize_model


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze the pretrained weights 在CPU上，这大约是之前方案的一半时间
    #for param in model_ft.parameters():
    #    param.requires_grad = False
    
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, loss_fn, optimizer_ft, scheduler, num_epochs=25)

    visualize_model(model_ft)

if __name__ == "__main__":
    main()