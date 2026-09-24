# LearnTheBasics

一个用于学习 PyTorch 基础流程的小项目。

项目使用 FashionMNIST 数据集训练一个全连接神经网络，包含数据下载、模型定义、训练和测试等基本步骤。

## 项目结构

```text
LearnTheBasics/
├── main.py             # 程序入口
├── mmModel.py          # 神经网络模型
├── train.py            # 训练逻辑
├── test.py             # 测试与准确率计算
└── download_data.py    # 下载数据集并创建 DataLoader
```

## 环境要求

- Python 3
- PyTorch
- torchvision

安装依赖：

```bash
pip install torch torchvision
```

## 运行方式

进入当前项目目录：

```bash
cd LearnTheBasics
python main.py
```

程序会自动下载 FashionMNIST 数据集，然后进行模型训练和测试。

## 模型结构

模型由以下部分组成：

1. 将 28×28 的图片展平
2. 全连接层：784 → 512
3. ReLU 激活函数
4. 全连接层：512 → 256
5. ReLU 激活函数
6. 输出层：256 → 10

输出的 10 个类别对应 FashionMNIST 的十种服装分类。