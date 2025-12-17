import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        return F.relu(out)

class Classifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.blocks = nn.Sequential(
            ResidualBlock(32, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.blocks(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def compute_ece(probs, targets, n_bins=100):
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(targets)
    
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.ge(bin_lower) & confidences.lt(bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()

def load_classifier(path, in_channels=1, device="cpu"):
    model = Classifier(in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="classifier.pth")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar"])
    args = parser.parse_args()

    print(f"Device: {args.device} | Dataset: {args.dataset}")

    if args.dataset == "mnist":
        in_channels = 1
        train_transforms_list = [
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.Normalize((0.5,), (0.5,)),
        ]
        val_transforms_list = [
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.Normalize((0.5,), (0.5,)),
        ]
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.Compose(train_transforms_list))
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.Compose(val_transforms_list))
    else:
        in_channels = 3
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = Classifier(in_channels=in_channels).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    wandb.init(project=f"{args.dataset}-classifier", config=args)

    print("Starting training")
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(1) == y).float().mean()
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"train_loss": loss.item(), "train_step_acc": acc.item(), "lr": current_lr})
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        scheduler.step()

        model.eval()
        val_loss = 0.0
        
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(args.device), y.to(args.device)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                
                probs = F.softmax(logits, dim=1)
                
                all_probs.append(probs)
                all_targets.append(y)

        all_probs = torch.cat(all_probs)
        all_targets = torch.cat(all_targets)

        val_acc = (all_probs.argmax(1) == all_targets).float().mean().item()
        avg_val_loss = val_loss / len(test_loader)
        val_ece = compute_ece(all_probs, all_targets)

        print(f"Val Acc: {val_acc:.4f} | Val ECE: {val_ece:.4f}")

        wandb.log({
            "val_acc": val_acc,
            "val_loss": avg_val_loss,
            "val_ece": val_ece,
            "epoch": epoch,
        })

    torch.save(model.state_dict(), args.save_path)

    wandb.finish()