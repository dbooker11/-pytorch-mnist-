import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

model = VisionTransformer(image_size=28, patch_size=4, num_classes=10, embed_dim=64, depth=6, heads=8, mlp_dim=128).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

train_model(model, trainloader, criterion, optimizer, epochs=30)
test_model(model, testloader)

