"""
Architecture: 6n+2 layers, n=3 → 20 layers
  - stem: 3×3 conv, 16 filters
  - 3 groups of n=3 blocks: 16, 32, 64 filters
  - Option A shortcuts: identity + zero-pad on dimension change
  - Global avg pool → FC(10)
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

n            = 3        # blocks per group → 6n+2 = 20 layers
EPOCHS       = 200
BATCH_SIZE   = 128
LR           = 0.1
MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-4
LR_DROPS     = [100, 150]
NUM_WORKERS  = 2

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Option A: identity shortcut with zero-padding for dimension change
        if self.stride != 1 or out.size(1) != x.size(1):
            x = torch.nn.functional.avg_pool2d(x, 1, stride=self.stride)
            pad = out.size(1) - x.size(1)
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, pad//2, pad//2))

        return self.relu(out + x)


class ResNet(nn.Module):
    def __init__(self, n, num_classes=10):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.relu   = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, 16, n, stride=1)
        self.layer2 = self._make_layer(16, 32, n, stride=2)
        self.layer3 = self._make_layer(32, 64, n, stride=2)
        self.fc     = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def _make_layer(self, in_planes, planes, n, stride):
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(n - 1):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)


def get_dataloaders():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_loader = DataLoader(
        torchvision.datasets.CIFAR10("./data", train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        torchvision.datasets.CIFAR10("./data", train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    )
    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct  += out.argmax(1).eq(y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct  += out.argmax(1).eq(y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


def plot_curves(history, save_path="curves.png"):
    epochs = range(1, len(history["tr_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["tr_loss"], label="Train")
    ax1.plot(epochs, history["te_loss"], label="Test")
    for ep in LR_DROPS:
        ax1.axvline(ep, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, [a * 100 for a in history["tr_acc"]], label="Train")
    ax2.plot(epochs, [a * 100 for a in history["te_acc"]], label="Test")
    for ep in LR_DROPS:
        ax2.axvline(ep, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curves saved to {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  ResNet-{6*n+2} (n={n})\n")

    train_loader, test_loader = get_dataloaders()
    model     = ResNet(n).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_DROPS, gamma=0.1)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}\n")
    print(f"{'Ep':>4}  {'LR':>6}  {'Tr Loss':>8}  {'Tr Acc':>7}  {'Te Loss':>8}  {'Te Acc':>7}  {'Time':>5}")
    print("-" * 62)

    best_acc = 0.0
    history  = {"tr_loss": [], "tr_acc": [], "te_loss": [], "te_acc": []}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = eval_epoch(model,  test_loader,  criterion, device)
        scheduler.step()

        history["tr_loss"].append(tr_loss)
        history["tr_acc"].append(tr_acc)
        history["te_loss"].append(te_loss)
        history["te_acc"].append(te_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:4d}  {lr:6.4f}  {tr_loss:8.4f}  {tr_acc*100:6.2f}%  "
              f"{te_loss:8.4f}  {te_acc*100:6.2f}%  {time.time()-t0:4.1f}s")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), f"resnet{6*n+2}_cifar10.pth")

    print(f"\nBest test accuracy: {best_acc*100:.2f}%")
    plot_curves(history)


if __name__ == "__main__":
    main()
