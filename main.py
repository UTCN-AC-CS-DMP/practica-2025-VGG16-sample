import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from vgg16 import VGG16


def get_data_loaders(data_dir, batch_size=64, valid_size=0.1, seed=42):
    transform = transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )
    
    print("📥 Downloading CIFAR-100 dataset...")

    full_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    print("📦 Dataset downloaded and transformed.")

    num_train = len(full_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.seed(seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        full_dataset, batch_size=batch_size, sampler=train_sampler
    )
    valid_loader = DataLoader(
        full_dataset, batch_size=batch_size, sampler=valid_sampler
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def train_model():
    
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )    

    train_loader, valid_loader, test_loader = get_data_loaders("./data", batch_size=64)    

    model = VGG16(num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.005
    )

    num_epochs = 20
    best_val_acc = 0.0
    
    print("🚀 Starting training...")
    print(f"🚀 Using device: {device}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Wrap your DataLoader with tqdm for progress bar
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        )

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=avg_loss)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_vgg16.pth")
            print("✅ Best model saved.")

    # Testing
    print("\n🧪 Evaluating best model on test set...")
    model.load_state_dict(torch.load("best_vgg16.pth"))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"🎯 Test Accuracy: {100 * correct / total:.2f}%")


# -------------------------
# Export model to ONNX
# -------------------------
def export_to_onnx(device, model):
    onnx_path = "vgg16_cifar100.onnx"
    dummy_input = torch.randn(1, 3, 227, 227).to(device)  # Match input shape

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,  # Store learned parameters
        opset_version=11,  # Commonly supported opset
        do_constant_folding=True,  # Fold constants for optimization
        input_names=["input"],
        output_names=["output"],
        # IMPORTANT: OpenCV's DNN module does not support dynamic axes with ONNX!
        # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"📦 Exported model to ONNX format: {onnx_path}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    train_model()
    # device = torch.device("cpu")
    # model = VGG16(num_classes=100)
    # model.load_state_dict(torch.load("best_vgg16.pth", map_location="cpu"))
    # model.eval()

    # export_to_onnx(device, model)
