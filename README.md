# -pytorch-mnist-
基于pytorch的mnist图像分类
# LeNet MNIST 分类项目
项目简介

本项目实现了一个基于 LeNet 架构的卷积神经网络，用于对 MNIST 手写数字数据集进行分类。使用 PyTorch 进行模型的构建、训练和评估。该网络由两个卷积层（conv1 和 conv2）、两个全连接层（fc1 和 fc2）以及一个输出层（fc3）组成。通过这个项目，你可以深入了解卷积神经网络（CNN）在图像分类中的应用，以及如何使用 PyTorch 进行神经网络的训练和测试。

代码说明

1.LeNet 类

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

2.训练函数

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
3.测试函数

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
4.数据加载与模型训练

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
5.输出示例

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
3.总结

本项目展示了如何使用 PyTorch 实现一个简单的 LeNet 网络，并用它对 MNIST 数据集进行分类。你可以根据需要调整超参数，如学习率、batch size、训练轮数等，来优化模型的性能。
# VGG MNIST图像分类项目
这是一个使用 PyTorch 实现的卷积神经网络（VGGNet）来对 MNIST 数据集进行手写数字分类的项目。代码定义了一个简单的 VGGNet 模型，并使用交叉熵损失和 Adam 优化器进行训练和测试。

项目简介

这个项目使用卷积神经网络（CNN）来分类 MNIST 数据集中的手写数字。网络架构借鉴了经典的 VGG 网络，包含多个卷积层和全连接层。通过训练模型，能够对测试集中的图像进行分类，最终输出模型的准确率。

使用方法

1. 数据准备

数据集使用 MNIST 数据集（手写数字数据集）。你可以通过 PyTorch 的 torchvision.datasets.MNIST 类来加载数据。代码默认从网络下载并存储到本地文件夹 ./data。若数据已经存在，download=False 将阻止重新下载数据。

2.训练模型

在 main.py 文件中，我们定义了一个 VGGNet 类，这是一个简化版本的 VGG 网络。网络结构如下：

卷积层 1：输入通道为 1（灰度图像），输出通道为 64，卷积核大小为 3x3，采用 ReLU 激活函数，并且进行了最大池化操作。
卷积层 2：输入通道为 64，输出通道为 128，同样使用 3x3 卷积核和 ReLU 激活函数。
卷积层 3：输入通道为 128，输出通道为 256，继续使用 3x3 卷积核和 ReLU 激活函数。
全连接层：三个全连接层，最终输出 10 个数字类别（0-9）。

训练过程通过调用 train_model 函数进行，该函数会在每个训练周期（epoch）输出当前的平均损失。

3. 测试模型

训练完成后，使用 test_model 函数对测试集进行评估。该函数计算并输出模型在测试集上的准确率。

代码说明

1. 数据预处理
数据加载使用 torchvision.transforms 对 MNIST 图像进行标准化处理
```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
transforms.ToTensor()：将图像转换为 PyTorch tensor 格式。
transforms.Normalize((0.5,), (0.5,))：将图像像素值归一化到 [-1, 1] 之间。
```
2. 定义模型

VGGNet 类继承自 nn.Module，并在其 __init__ 构造函数中定义了卷积层和全连接层。模型的前向传播通过 forward 函数实现
```python
def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = x.view(x.size(0), -1)  # Flatten
    x = nn.ReLU()(self.fc1(x))
    x = nn.ReLU()(self.fc2(x))
    x = self.fc3(x)
    return x
````
3. 训练和优化

使用 Adam 优化器和 CrossEntropyLoss 作为损失函数，训练模型并计算损失值：
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```
  train_model 函数会在每个 epoch 输出当前的训练损失：
```python
running_loss += loss.item()
```
4. 测试和评估
```python   
test_model 函数在测试集上评估模型的准确性：
correct += (predicted == labels).sum().item()

最终输出模型在测试集上的分类准确率：
print(f'Accuracy of the vgg model: {100 * correct / total:.2f}%')
```
5.输出示例
```python
Epoch 1, Loss: 0.21931541252824688
Epoch 2, Loss: 0.08156749702160097

...
Epoch 30, Loss: 0.004073647625465222

Accuracy of the vgg model: 99.45%
```
总结

本项目展示了如何使用 PyTorch 实现一个基于 VGG 网络的图像分类模型，应用于 MNIST 手写数字数据集。通过简单的卷积层和全连接层，模型能够准确地对数字进行分类。

# ResNet MNIST图像分类项目
该项目实现了一个基于ResNet架构的深度学习模型，用于MNIST数据集上的手写数字分类。以下是对代码的详细说明，包括如何训练和测试模型，以及运行结果的示例。

项目结构

train_model：用于训练模型的函数。
test_model：用于评估模型在测试集上的性能的函数。
BasicBlock：ResNet的基本模块，用于构建残差连接。
ResNet：实现ResNet网络结构。

代码说明

1. 训练函数 train_model
```python
def train_model(model, trainloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')
```
train_model 函数用于训练模型，接受模型、训练数据加载器、损失函数、优化器和训练轮数（默认为10）作为输入参数。
每一个epoch，模型会根据训练数据进行前向传播、计算损失、反向传播并更新参数，最后打印出该epoch的平均损失。

2. 测试函数 test_model
```python
def test_model(model, testloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the vgg model: {100 * correct / total:.2f}%')
```
test_model 函数用于评估模型在测试集上的准确率。通过与真实标签进行对比，计算模型在测试集上的预测正确率。

3. ResNet模型架构
```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out
```
BasicBlock 是ResNet中的核心模块，包含两层卷积和一个快捷连接（shortcut）。如果输入和输出的尺寸不匹配，会通过1x1卷积调整尺寸，使得输入和输出可以相加。
```python
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # 修改这里
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
```

ResNet 类定义了整个网络结构，包括初始卷积层、四个残差块（layer1, layer2, layer3, layer4），以及最终的全连接层。

4. 数据预处理和加载
```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
```
数据加载部分使用torchvision.datasets.MNIST加载MNIST数据集，并应用ToTensor和Normalize转换，以便将图像转换为Tensor并进行标准化。

5. 模型训练与测试
```python
model = ResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, trainloader, criterion, optimizer, epochs=30)
test_model(model, testloader)
```
在这里，我们创建了一个ResNet模型实例，并指定损失函数为交叉熵损失，优化器为Adam优化器。然后，调用train_model进行训练，并通过test_model评估模型的测试集准确率。

6. 输出示例
```python
假设训练了30个epoch，输出的示例如下：
Epoch 1, Loss: 0.2113
Epoch 2, Loss: 0.1042
Epoch 3, Loss: 0.0759
...
Epoch 30, Loss: 0.0178
Accuracy of the vgg model: 98.32%
```
总结 

本项目实现了一个使用ResNet架构的深度学习模型，成功地应用于MNIST手写数字分类任务。通过残差连接的方式，模型能够更好地进行训练，尤其在较深的网络中，避免了梯度消失问题。
# Vision Transformer (ViT) MNIST 分类模型
本项目实现了一个基于 Vision Transformer (ViT) 的图像分类模型，使用 PyTorch 框架在 MNIST 数据集上进行训练和测试。该模型将 MNIST 图像分成多个小块（patches），通过 Transformer 架构进行特征提取，并最终进行分类。

代码结构

1. 定义模型

PatchEmbedding 类

该类用于将输入图像拆分成多个小块（patches）并进行嵌入，生成一个嵌入矩阵。
```python
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=28, patch_size=4, num_channels=1, embed_dim=64):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  
        x = x.flatten(2)  
        x = x.transpose(1, 2)  
        return x
```

TransformerBlock 类

该类定义了 Transformer 的基本模块，包括多头自注意力层和前馈神经网络。
```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, mlp_dim, drop_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(drop_rate),
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        mlp_output = self.mlp(x)
        x = self.layer_norm2(x + mlp_output)
        return x
```
每个 Transformer 块包括多头自注意力机制和一个前馈神经网络，且采用层归一化进行稳定训练。


VisionTransformer 类

该类是整个模型的主体，包括了 PatchEmbedding、位置编码、Transformer 块堆叠和最终的分类层。
```python
class VisionTransformer(nn.Module):
    def __init__(self, image_size=28, patch_size=4, num_classes=10, embed_dim=32, depth=8, heads=4, mlp_dim=64):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, num_channels=1, embed_dim=embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2, embed_dim))
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, heads, mlp_dim) for _ in range(depth)]
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        patches = self.patch_embedding(x)
        x = patches + self.pos_embedding
        x = self.transformer_blocks(x)
        x = x.mean(dim=1)  # 全局平均池化代替 cls_token
        x = self.fc(x)
        return x
```

2. 数据预处理

MNIST 数据集使用 torchvision.transforms.Compose 进行预处理，包括标准化和转换为张量。
```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
```
3. 模型训练和测试

train_model 
```python
def train_model(model, trainloader, criterion, optimizer, epochs=30):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}')
```
test_model 
```
def test_model(model, testloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the vit model: {100 * correct / total:.2f}%')
```
```python
train_model(model, trainloader, criterion, optimizer, epochs=30)
test_model(model, testloader)
```
```python
输出示例
训练过程中的输出示例：
Epoch 1, Loss: 0.5568
Epoch 2, Loss: 0.3377
Epoch 3, Loss: 0.2713
...
Epoch 30, Loss: 0.0234

测试过程中的输出示例：
Accuracy of the vit model: 98.23%
```


总结

该模型使用 Vision Transformer (ViT) 对 MNIST 数据集进行分类，通过卷积层将图像分割为多个小块，并通过 Transformer 架构进行特征提取。通过训练，模型能够在 MNIST 测试集上达到较高的准确率。
