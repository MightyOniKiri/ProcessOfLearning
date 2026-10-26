import time
import os
import torch
import matplotlib.pyplot as plt
import random
import csv

from load_data import class_names, dataloaders, dataset_sizes, device, imshow

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    history = []

    # 最佳权重保存在项目目录，训练结束后文件仍然保留
    best_model_params_path = os.path.join(os.path.dirname(__file__), 'best_model_params.pt')
    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_result = {'epoch': epoch + 1}
        
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #model.eval()
                #model.fc.train()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                # 只在训练阶段清空梯度
                if phase == 'train':
                    optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            epoch_result[f'{phase}_loss'] = epoch_loss
            epoch_result[f'{phase}_acc'] = epoch_acc.item()
            
            if phase == 'val':
                history.append(epoch_result)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights for evaluation
    model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    
    # 从历史记录中取出轮数、训练损失和验证损失
    epochs = [result['epoch'] for result in history]
    train_losses = [result['train_loss'] for result in history]
    val_losses = [result['val_loss'] for result in history]

    # 创建一张图，画出两条曲线
    plt.figure()
    plt.plot(epochs, train_losses, label='train')
    plt.plot(epochs, val_losses, label='val')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.legend()

    # 保存到项目目录
    save_path = os.path.join(os.path.dirname(__file__), 'loss_curve.png')
    plt.savefig(save_path)
    plt.close()
    
    # 取出每轮训练和验证的准确率
    train_accs = [result['train_acc'] for result in history]
    val_accs = [result['val_acc'] for result in history]

    plt.figure()
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='val')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    save_path = os.path.join(
            os.path.dirname(__file__),
            'accuracy_curve.png'
        )
    plt.savefig(save_path)
    plt.close()
    
    csv_path = os.path.join(
        os.path.dirname(__file__),
        'training_history.csv'
    )

    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'epoch',
                'train_loss',
                'train_acc',
                'val_loss',
                'val_acc',
            ],
        )

        writer.writeheader()
        writer.writerows(history)
    
    return model
                    
                    
def visualize_model(model, num_images=6, save_path=None):
    was_training = model.training
    model.eval()
    image_so_far = 0
    fig = plt.figure(figsize=(8, 3 * ((num_images + 1) // 2)))

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'prediction_results.png')

    def save_figure():
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Prediction image saved to: {os.path.abspath(save_path)}')
        
    dataset = dataloaders['val'].dataset
    rng = random.Random(42)
    selected_indices = []

    for class_index in range(len(class_names)):
        # 找出属于当前类别的图片编号
        candidates = [
            index
            for index, label in enumerate(dataset.targets)
            if label == class_index
        ]

        # 每类固定随机选 3 张
        selected_indices.extend(rng.sample(candidates, 3))

    display_dataset = torch.utils.data.Subset(
        dataset, selected_indices
    )

    display_loader = torch.utils.data.DataLoader(
        display_dataset,
        batch_size=4,
        shuffle=False,
    )
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(display_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                image_so_far += 1
                ax = plt.subplot((num_images + 1)//2, 2, image_so_far)
                ax.axis('off')
                predicted_index = preds[j].item()
                true_index = labels[j].item()
                if predicted_index == true_index:
                    title_color = 'green'
                else:
                    title_color = 'red'
                ax.set_title(
                    f'predicted: {class_names[predicted_index]}\n'
                    f'true: {class_names[true_index]}',
                    color=title_color
                )
                
                imshow(inputs[j].cpu())
                 
                if image_so_far == num_images:
                    save_figure()
                    model.train(mode=was_training)
                    return save_path
        save_figure()
        model.train(mode=was_training)
        return save_path
                    
