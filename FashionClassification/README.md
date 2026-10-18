# FashionClassification

> 项目参考来源：PyTorch 官方 [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) 系列教程，主要对应 [Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) 中的 FashionMNIST 分类示例。本项目在学习示例的基础上拆分代码模块、调整网络层宽度，并增加训练指标记录、数据可视化及测试样例保存功能。目前已将数据集切换为 EMNIST Balanced，输出层相应调整为 47 类。

一个用于学习 PyTorch 基础流程的小项目。

项目目前使用 EMNIST Balanced 数据集训练一个手写字符分类的全连接神经网络，包含数据下载、模型定义、训练、测试和可视化等步骤。文件夹保留 `FashionClassification` 名称。

## 当前数据集：EMNIST Balanced

数据集来源：[NIST 官方 EMNIST 介绍](https://www.nist.gov/itl/products-and-services/emnist-dataset)。

- 输入为 28×28 的单通道灰度图片，包含手写数字和字母。
- 当前使用 `split="balanced"`，共 47 个类别，标签索引为 0～46。
- 训练集通过 `train=True` 加载，测试集通过 `train=False` 加载；两者均使用 `split="balanced"`。
- `split` 在这里选择 EMNIST 的版本，`train` 才用于选择该版本的训练集或测试集。
- 图像经过 `v2.ToImage()` 和 `v2.ToDtype(torch.float32, scale=True)` 转换为浮点张量，像素值范围为 0～1。
- 可视化中的字符名称取自数据集的 `classes` 属性。

加载配置示例（与 `download_data.py` 一致）：

```python
train_dataset = datasets.EMNIST(
    root="./data",
    split="balanced",
    train=True,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]),
)
```

测试集使用相同配置，但将 `train` 改成 `False`。`root="./data"` 相对于运行命令时所在的目录，因此建议按下面的方式进入项目目录后运行。

## 项目结构

```text
FashionClassification/
├── main.py             # 程序入口
├── mmModel.py          # 神经网络模型
├── train.py            # 训练逻辑
├── test.py             # 测试与准确率计算
├── download_data.py    # 下载数据集并创建 DataLoader
├── visualize.py        # 新增：保存样例图片、预测记录和训练曲线
├── predict.py          # 新增：加载已有模型并保存测试样例预测
├── model.pth           # 训练后保存的模型参数（不提交到 Git）
└── outputs/            # 生成的图片和 CSV（不提交到 Git）
```

## 环境要求

- Python 3
- PyTorch
- torchvision
- matplotlib

安装依赖：

```bash
pip install torch torchvision matplotlib
```

## 运行方式

进入当前项目目录：

```bash
cd FashionClassification
python main.py
```

程序会自动下载 EMNIST 数据集并加载 Balanced 版本，然后进行模型训练和测试。

从原来的 FashionMNIST 切换后，需要重新运行 `python main.py` 训练模型。旧的 10 类 `model.pth` 无法直接加载到当前 47 类网络；如需保留旧参数，请先备份。`python predict.py` 应使用当前 EMNIST 网络训练得到的参数。

## 模型结构

模型由以下部分组成：

1. 将 28×28 的图片展平
2. 全连接层：784 → 512
3. ReLU 激活函数
4. 全连接层：512 → 256
5. ReLU 激活函数
6. 输出层：256 → 47

输出的 47 个分数（logits）对应 EMNIST Balanced 的 47 个字符类别。训练时将 logits 直接传给 `CrossEntropyLoss`，预测展示时再用 softmax 计算各类别概率。

## 可视化与测试样例保存

运行 `python main.py` 后，项目目录的 `outputs/` 中会生成：

- `data_samples.png`：16 张训练数据样例及真实类别。
- `training_curves.png`：每轮训练和测试的平均损失、准确率（0～1）。
- `history.csv`：每轮的指标，便于自己绘图分析。
- `test_predictions.png`：16 张测试图片，标注真实类别、预测类别和预测概率；绿色表示正确，红色表示错误。
- `test_predictions.csv`：对应样例的索引、标签、概率、是否正确和图片路径（相对于 outputs 目录）。
- `sample_images/`：对应测试样例的单独灰度 PNG 图片。

已有当前网络训练得到的 `model.pth` 时，进入本目录运行 `python predict.py`，即可加载参数并生成测试预测图、CSV 和样例图片，不会重新训练。第一次使用仍需要下载数据集。

`visualize.py` 中的 `save_samples()` 负责样例可视化和保存，`save_history()` 负责曲线和指标保存。可通过 `count` 参数调整样例数量；固定随机种子使每次选择的样例一致。当前图片显示逻辑针对本项目的单通道、像素范围 0～1 的图片。

注意：图中概率是模型 softmax 输出，不代表经过校准的可信度；训练指标来自训练过程中各批次，测试指标来自该轮训练结束后的模型。测试曲线仅用于本项目学习观察，如果据此选择模型或调整超参数，应另外划分验证集，保留测试集用于最终评估。

## 在原有代码基础上增加了什么

原来的 FashionMNIST 项目已经可以训练、测试并保存模型参数，但训练和测试指标只打印在终端，没有保存样例图片、预测记录或训练曲线。后来增加了下面的记录和可视化功能；当前又切换到 EMNIST Balanced，网络结构相应更新为 `784 → 512 → 256 → 47`。

### 1. train.py：统计并返回整轮训练指标

训练函数新增总损失、预测正确数量和样本数量的累计：

```python
total_loss += loss.item() * labels.size(0)
correct += (pred_labels.argmax(1) == labels).sum().item()
total_samples += labels.size(0)
```

- `labels.size(0)` 是当前批次的样本数。
- `pred_labels.argmax(1)` 为每张图片选出分数最高的类别。
- 与真实标签比较后，`sum().item()` 得到当前批次预测正确的数量。
- `CrossEntropyLoss` 默认返回批次平均损失，乘以批次样本数后，才能累计为样本损失之和。

一轮结束后返回字典，供主程序记录和绘图：

```python
return {
    "loss": total_loss / total_samples,
    "accuracy": correct / total_samples,
}
```

这些统计不参与反向传播，原来的参数更新流程保持不变。

### 2. test.py：返回测试指标，修正平均损失计算

原来的测试损失按批次取平均，最后一个批次即使样本较少，也占同样权重。现在改为按样本数累计：

```python
total_loss += loss_fn(pred_labels, labels).item() * labels.size(0)
# 所有批次结束后
total_loss /= total_samples
```

测试函数仍使用 `model.eval()` 和 `torch.no_grad()`，同时新增返回值：

```python
return {"loss": total_loss, "accuracy": accuracy}
```

### 3. visualize.py：新增样例保存与绘图模块

#### save_samples()：展示图片，保存预测结果

函数接口：

```python
save_samples(dataset, output_path, model=None, device="cpu", count=16)
```

不传模型时，只显示图片及真实类别，用于观察数据：

```python
save_samples(train_loader.dataset, output_dir / "data_samples.png")
```

传入模型时，先对选中的图片预测，再保存预测图、同名 CSV 和单独的样例图片：

```python
save_samples(
    test_loader.dataset,
    output_dir / "test_predictions.png",
    model,
    device,
)
```

预测的核心代码：

```python
probabilities = model(images.to(device)).softmax(dim=1)
confidences, predictions = probabilities.max(dim=1)
```

模型输出的是 logits；`softmax(dim=1)` 将每张图片的各类别分数转换为概率，`max(dim=1)` 返回最大概率及其类别索引。softmax 仅用于这里的结果展示，训练时仍将 logits 直接交给 `CrossEntropyLoss`。

每张图会显示真实类别、预测类别及预测概率。正确预测使用绿色标题，错误预测使用红色标题。CSV 记录数据集索引、真实标签和类别、预测标签和类别、预测概率、是否正确以及图片路径。

函数使用固定随机种子 `42` 抽取样例，默认抽取 16 张，便于在不同训练结果之间比较相同图片。预测期间临时切换到评估模式并关闭梯度计算，结束后恢复模型原来的训练状态。

图片通过 matplotlib 的 `Agg` 后端直接保存，不会弹出绘图窗口。单张样例图片保存为 PNG，CSV 通过图片路径与之对应，并没有将测试集重新保存为 PyTorch 数据集文件。

#### save_history()：保存指标与训练曲线

```python
save_history(history, output_dir)
```

`history` 是一个列表，每个元素记录一轮的 epoch、训练损失、训练准确率、测试损失和测试准确率。函数将其保存为 `history.csv`，并绘制损失与准确率两张子图，合并保存到 `training_curves.png`。

### 4. main.py：接入记录和可视化流程

主程序增加的调用顺序如下：

```text
保存训练数据样例图
    ↓
每轮训练，获取训练指标
    ↓
每轮测试，获取测试指标
    ↓
将本轮指标加入 history，更新 CSV 和曲线
    ↓
全部训练结束，保存模型参数
    ↓
保存测试预测图、预测 CSV 和单独的测试图片
```

每轮的指标这样组织：

```python
history.append({
    "epoch": turn + 1,
    "train_loss": train_metrics["loss"],
    "train_accuracy": train_metrics["accuracy"],
    "test_loss": test_metrics["loss"],
    "test_accuracy": test_metrics["accuracy"],
})
```

模型参数固定保存到 `main.py` 所在目录的 `model.pth`，可视化结果固定保存到同目录的 `outputs/`。再次运行会覆盖这些同名输出；如需保留不同实验的结果，请先复制或重命名输出目录及模型文件。

### 5. predict.py：加载现有模型，单独生成预测样例

新增入口用于已有 `model.pth` 时直接查看模型预测，无需重新训练：

```bash
cd FashionClassification
python predict.py
```

核心流程为：创建相同结构的网络 → 加载参数 → 调用 `save_samples()`。

```python
model.load_state_dict(
    torch.load(checkpoint, map_location=device, weights_only=True)
)
```

`map_location` 指定加载参数所用的设备，`weights_only=True` 用于加载参数文件。此入口默认选择可用的 CUDA，否则使用 CPU；参数文件位于 `predict.py` 所在目录，必须与当前网络结构兼容。

此脚本只对抽取的样例进行预测并保存，不会计算整个测试集的准确率。它仍使用现有的数据加载模块，导入时会创建训练集和测试集，缺少数据时会尝试下载。

### 6. README.md 与根目录 .gitignore

- README 新增 matplotlib 依赖、运行说明、输出说明和本节代码讲解。
- 根目录 `.gitignore` 新增 `FashionClassification/outputs/`，避免把生成的图片和 CSV 提交到 Git。模型参数文件已由原来的 `*.pth` 规则忽略。

### 验证情况

本次修改已通过项目中 7 个 Python 文件的语法检查。修改时可用的 Python 环境缺少 torch、torchvision 和 matplotlib，因此尚未实际运行训练、加载模型及绘图流程；需要在安装好上述依赖的环境中运行验证。
