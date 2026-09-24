# ImageClassification

这是一个使用 PyTorch 学习图像分类和迁移学习的练习项目，参考
[Transfer Learning for Computer Vision Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial)。

当前阶段主要实现：

- 使用 `ImageFolder` 读取训练集和验证集
- 对训练图片进行随机裁剪和水平翻转
- 使用 ImageNet 的均值和标准差对图片进行标准化
- 使用 `DataLoader` 分批加载数据
- 显示一个批次的图片及其类别

## 项目结构

```text
ImageClassification/
├── README.md
├── load_data.py
└── data/
    └── hymenoptera_data/
        ├── train/
        │   ├── ants/
        │   └── bees/
        └── val/
            ├── ants/
            └── bees/
```

`data/` 目录已被 Git 忽略，不会上传到仓库。

## 环境要求

- Python 3
- PyTorch
- torchvision
- Matplotlib
- NumPy

安装依赖：

```bash
pip install torch torchvision matplotlib numpy
```

## 准备数据集

下载 PyTorch 教程使用的 Hymenoptera 数据集，并解压到：

```text
ImageClassification/data/hymenoptera_data/
```

程序要求数据集包含 `train`、`val` 两部分，并在它们下面按类别建立文件夹。

## 运行方式

进入当前项目目录后运行：

```bash
cd ImageClassification
python load_data.py
```

程序会输出当前使用的计算设备，并显示一个批次的训练图片和对应类别。

## 后续计划

- 加载在 ImageNet 上预训练的模型
- 替换并训练分类层
- 比较微调整个网络与固定特征提取器的效果
- 在验证集上评估模型准确率
