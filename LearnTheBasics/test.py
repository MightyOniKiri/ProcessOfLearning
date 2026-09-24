import torch

def test(test_loader, model, loss_fn, device):
    num_batches = len(test_loader)
    # 调整模型为评估格式
    model.eval()  
    
    total_loss = 0
    correct_pred = 0
    total_samples = 0
    
    with torch.no_grad():  # 在测试阶段不需要计算梯度
        for images, labels in test_loader:
            # 测试数据也必须和模型处于同一个设备。
            images, labels = images.to(device), labels.to(device)
            
            pred_labels = model(images)
            total_loss += loss_fn(pred_labels, labels).item()
            correct_pred += (pred_labels.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
            
    total_loss /= num_batches
    accuracy = correct_pred / total_samples
    
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {total_loss:>8f} \n")
