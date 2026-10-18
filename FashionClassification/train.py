import torch
import torch.nn as nn

def train(train_loader, model, loss_fn, optimizer, device):
    size = len(train_loader.dataset)
    # 调整模型为训练格式
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    for batch, (images, labels) in enumerate(train_loader):
        # DataLoader 留在 CPU；只把当前批次的数据移动到模型所在设备。
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播，本质就是让数据经过model的一层层网络，得到输出结果pred，就是预测的labels
        pred_labels = model(images)
        loss = loss_fn(pred_labels, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (pred_labels.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)
        
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新参数
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return {"loss": total_loss / total_samples, "accuracy": correct / total_samples}
            
