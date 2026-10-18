import torch
from pathlib import Path

from train import train
from test import test
from mmModel import NeuralNetwork
from download_data import train_loader, test_loader
from visualize import save_samples, save_history

def main():
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    output_dir = Path(__file__).resolve().parent / "outputs"
    save_samples(train_loader.dataset, output_dir / "data_samples.png")
    
    model = NeuralNetwork().to(device)
    # model.load_state_dict(torch.load("model.pth", weights_only=True))
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    epochs = 5
    history = []
    for turn in range(epochs):
        print(f"Epoch {turn+1}\n-------------------------------")
        # DataLoader 本身不能移动到设备；把 device 交给训练和测试函数，
        # 由它们在取出每一批数据后移动 images 和 labels。
        train_metrics = train(train_loader, model, loss_fn, optimizer, device)
        test_metrics = test(test_loader, model, loss_fn, device)
        history.append({"epoch": turn + 1, "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                        "test_loss": test_metrics["loss"], "test_accuracy": test_metrics["accuracy"]})
        save_history(history, output_dir)
        
    torch.save(model.state_dict(), Path(__file__).resolve().parent / "model.pth")     #state_dict() 只保存模型的参数，是一个字典
    print("Saved PyTorch Model State to model.pth")
    save_samples(test_loader.dataset, output_dir / "test_predictions.png", model, device)
    print(f"Saved figures and test samples to {output_dir}")

if __name__ == "__main__":
    main()
