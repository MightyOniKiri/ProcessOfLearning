import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision

cudnn.benchmark = True
plt.ion()   # interactive mode

data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224),      # 随机选取图片的一部分并缩放为224*224大小，这样可以增加数据的多样性
        transforms.RandomHorizontalFlip(),      # 以一定概率左右翻转图片，默认概率是 50%。这也是一种数据增强
        transforms.ToTensor(),
        
        # RGB 三个通道进行标准化，计算方式是：
        # 标准化后的像素 = (原像素 - 均值) / 标准差
        # 这些数值是 ImageNet 数据集常用的均值和标准差，适合搭配在 ImageNet 上预训练过的 ResNet 等模型。
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = './data/hymenoptera_data'
# read all images and save them to image_datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
# create dataloaders 可以理解为一个批量数据加载器
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 4,
                                              shuffle = True, num_workers = 4)
                  for x in ['train', 'val']}
# 统计每个数据集的大小
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# 统计类别名称
class_names = image_datasets['train'].classes

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"



# Visualizing a few images
def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def main():
    print(f"Using device: {device}")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
