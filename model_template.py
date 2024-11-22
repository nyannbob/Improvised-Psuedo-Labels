import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import timm
import matplotlib.pyplot as plt
import warnings
import numpy as np
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm  # For progress bar

print(torch.cuda.is_available())
warnings.filterwarnings("ignore")

# Define helper functions
def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes

def get_data_loaders(data_dir, batch_size, train=False):
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter()], p=0.1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        all_data = datasets.ImageFolder(data_dir, transform=transform)
        train_data_len = int(len(all_data) * 0.75)
        valid_data_len = int((len(all_data) - train_data_len) / 2)
        test_data_len = len(all_data) - train_data_len - valid_data_len
        train_data, val_data, test_data = random_split(
            all_data, [train_data_len, valid_data_len, test_data_len]
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        return train_loader, train_data_len
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        all_data = datasets.ImageFolder(data_dir, transform=transform)
        train_data_len = int(len(all_data) * 0.70)
        valid_data_len = int((len(all_data) - train_data_len) / 2)
        test_data_len = len(all_data) - train_data_len - valid_data_len
        train_data, val_data, test_data = random_split(
            all_data, [train_data_len, valid_data_len, test_data_len]
        )
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
        return val_loader, test_loader, valid_data_len, test_data_len

# Dataset paths and loaders
dataset_path = "CUB_200_2011/CUB_200_2011/images"
train_loader, train_data_len = get_data_loaders(dataset_path, 32, train=True)
val_loader, test_loader, valid_data_len, test_data_len = get_data_loaders(dataset_path, 64, train=False)
classes = get_classes(dataset_path)

dataloaders = {"train": train_loader, "val": val_loader}
dataset_sizes = {"train": train_data_len, "val": valid_data_len}
print("Train data length: ", train_data_len, "Test data length: ", test_data_len, "Validation data length: ", valid_data_len)

# Visualization
dataiter = iter(train_loader)
images, labels = dataiter.__next__()
images = images.numpy()

fig = plt.figure(figsize=(25, 4))
for idx in range(20):
    ax = fig.add_subplot(2, 10, idx + 1)
    img = np.transpose(images[idx], (1, 2, 0))
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(classes[labels[idx]])
    ax.axis('off')
plt.tight_layout()
plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Define the model
def get_model(num_classes):
    model = timm.create_model('efficientnet_b0', pretrained=True)  # Use EfficientNet-B0
    model.classifier = nn.Linear(model.get_classifier().in_features, num_classes)  # Adjust the classifier layer
    return model.to(device)

# Train the model
def train_model(model, dataloaders, criterion, optimizer, num_epochs=2):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(dataloaders["train"], desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / dataset_sizes["train"]
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    return model

# Evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Main script for training and evaluation
if __name__ == '__main__':
    baseline_model = get_model(num_classes=len(classes))
    criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.Adam(baseline_model.parameters(), lr=1e-4)

    print("Training Baseline Model...")
    baseline_model = train_model(baseline_model, dataloaders, criterion, optimizer, num_epochs=2)
    baseline_accuracy = evaluate_model(baseline_model, test_loader)
