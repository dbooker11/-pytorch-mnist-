# -pytorch-mnist-
基于pytorch的mnist图像分类
# LeNet MNIST 分类项目
1.项目简介

本项目实现了一个基于 LeNet 架构的卷积神经网络，用于对 MNIST 手写数字数据集进行分类。使用 PyTorch 进行模型的构建、训练和评估。该网络由两个卷积层（conv1 和 conv2）、两个全连接层（fc1 和 fc2）以及一个输出层（fc3）组成。通过这个项目，你可以深入了解卷积神经网络（CNN）在图像分类中的应用，以及如何使用 PyTorch 进行神经网络的训练和测试。

2.环境要求

torch (PyTorch);torchvision;numpy

3.代码说明

LeNet 类

LeNet 类继承自 nn.Module，实现了一个简单的卷积神经网络结构。网络包含两个卷积层（conv1 和 conv2），以及三个全连接层（fc1, fc2, 和 fc3）。在 forward 方法中，定义了模型的前向传播过程。
```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)  # 展平
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

conv1: 输入通道为 1（灰度图），输出通道为 6，卷积核大小为 5x5。
conv2: 输入通道为 6，输出通道为 16，卷积核大小为 5x5。
fc1、fc2、fc3: 全连接层，逐步将特征向量从 16x5x5 转换为 120、84，再到 10（最终分类结果）。

训练函数

train_model 函数实现了训练过程。它接收模型、数据加载器、损失函数、优化器等作为输入，训练模型并输出每个 epoch 的平均损失。
```python
def train_model(model, trainloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')
```
测试函数
test_model 函数用来评估训练好的模型的准确度。它通过 torch.no_grad() 上下文管理器禁用梯度计算，避免了在测试过程中不必要的内存占用和计算。
```python
def test_model(model, testloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the LeNet model: {100 * correct / total:.2f}%')
```
数据加载与模型训练

在 main.py 文件中，我们加载了 MNIST 数据集并进行了训练与测试。通过以下代码，我们首先进行数据预处理，然后创建数据加载器（trainloader 和 testloader），并初始化模型、损失函数和优化器。
```python
if __name__ == "__main__":
    print("start....")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    print("加载数据...")
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("开始训练...")
    train_model(model, trainloader, criterion, optimizer, epochs=10)

    print("开始测试...")
    test_model(model, testloader)
```
输出

在运行过程中，你将看到每个 epoch 的训练损失和最终模型在测试集上的准确率。输出示例如下：
```python
start....
加载数据...
Before loading dataset...
After loading dataset...
训练集样本： 60000 torch.Size([60000, 28, 28])
开始训练...
Epoch 1, Loss: 0.2445
Epoch 2, Loss: 0.0821
...
开始测试...
Accuracy of the LeNet model: 98.11%
```
4.总结

本项目展示了如何使用 PyTorch 实现一个简单的 LeNet 网络，并用它对 MNIST 数据集进行分类。你可以根据需要调整超参数，如学习率、batch size、训练轮数等，来优化模型的性能。
