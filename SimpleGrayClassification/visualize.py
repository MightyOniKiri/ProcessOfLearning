"""保存数据样例、预测结果和训练曲线；无需打开图形窗口。"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def save_samples(dataset, output_path, model=None, device="cpu", count=16):
    """选取固定随机样例；提供模型时，同时保存预测 CSV 和原始灰度图片。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(count, len(dataset))
    if count < 1:
        raise ValueError("样例数量必须大于零，且数据集不能为空")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:count].tolist()
    samples = [dataset[index] for index in indices]
    images = torch.stack([image for image, _ in samples])
    predictions = confidences = None
    if model is not None:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                probabilities = model(images.to(device)).softmax(dim=1)
                confidences, predictions = probabilities.max(dim=1)
                confidences, predictions = confidences.cpu(), predictions.cpu()
        finally:
            model.train(was_training)

    columns = min(4, count)
    fig, axes = plt.subplots(math.ceil(count / columns), columns,
                             figsize=(3.5 * columns, 3.4 * math.ceil(count / columns)), squeeze=False)
    rows = []
    classes = dataset.classes
    image_dir = output_path.parent / "sample_images"
    if predictions is not None:
        image_dir.mkdir(exist_ok=True)
    for position, (image, label) in enumerate(samples):
        label = int(label)
        ax = axes.flat[position]
        pixels = image.squeeze(0).cpu().numpy()
        ax.imshow(pixels, cmap="gray", vmin=0, vmax=1)
        title = f"True: {classes[label]}"
        if predictions is not None:
            predicted = int(predictions[position])
            confidence = float(confidences[position])
            title += f"\nPred: {classes[predicted]} ({confidence:.1%})"
            filename = f"sample_{indices[position]}.png"
            plt.imsave(image_dir / filename, pixels, cmap="gray", vmin=0, vmax=1)
            rows.append([indices[position], label, classes[label], predicted,
                         classes[predicted], confidence, label == predicted, f"sample_images/{filename}"])
            ax.set_title(title, color="green" if label == predicted else "red", fontsize=9)
        else:
            ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[count:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    if rows:
        with output_path.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["dataset_index", "true_label", "true_class", "pred_label",
                             "pred_class", "confidence", "correct", "image_path"])
            writer.writerows(rows)


def save_history(history, output_dir):
    """history 的每一项保存一个 epoch 的训练和测试指标。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "test"):
            ax.plot(epochs, [row[f"{split}_{metric}"] for row in history], marker="o", label=split)
        ax.set(xlabel="Epoch", ylabel=metric.capitalize())
        ax.set_xticks(epochs)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)
