import torch

from train import train
from test import test
from mmModel import NeuralNetwork
from download_data import train_loader, test_loader

def main():
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    model = NeuralNetwork().to(device)
    # model.load_state_dict(torch.load("model.pth", weights_only=True))
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    epochs = 5
    for turn in range(epochs):
        print(f"Epoch {turn+1}\n-------------------------------")
        # DataLoader 本身不能移动到设备；把 device 交给训练和测试函数，
        # 由它们在取出每一批数据后移动 images 和 labels。
        train(train_loader, model, loss_fn, optimizer, device)
        test(test_loader, model, loss_fn, device)
        
    torch.save(model.state_dict(), "model.pth")     #state_dict() 只保存模型的参数，是一个字典
    print("Saved PyTorch Model State to model.pth")

if __name__ == "__main__":
    main()
