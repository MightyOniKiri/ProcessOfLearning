import torch
from pathlib import Path
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# transforms是 PyTorch 生态中专门用于图像预处理和数据增强（Data Augmentation）的模块
# v2是新版API，支持更多类型的输入，例如图像、目标检测中的边界框以及分割任务中的掩码，并能对相关数据进行协调变换
from torchvision.transforms import v2

# In this code, we use the EMNIST dataset from torchvision.datasets. 

# Download the trining and test datasets
data_dir = Path(__file__).resolve().parent / "data"
train_dataset = datasets.EMNIST(
    root = data_dir,
    split = "balanced",
    train = True,
    download = True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)
test_dataset = datasets.EMNIST(
    root = data_dir,
    split = "balanced",
    train = False,
    download = True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

batch_size = 64

#create data loaders
# shuffle=True 可以打乱数据
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

for images, labels in  test_loader:
    print(f"Shape of X [N, C, H, W]: {images.shape}")       #N是批次（batch）大小，C是通道数，H是高度，W是宽度
    print(f"Shape of y: {labels.shape} {labels.dtype}")
    break
