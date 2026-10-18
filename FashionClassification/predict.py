"""加载已经训练好的参数并保存测试样例，不重新训练。"""
from pathlib import Path

import torch

from download_data import test_loader
from mmModel import NeuralNetwork
from visualize import save_samples


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork().to(device)
    checkpoint = Path(__file__).resolve().parent / "model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    output_path = checkpoint.parent / "outputs" / "test_predictions.png"
    save_samples(test_loader.dataset, output_path, model, device)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
