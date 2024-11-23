import random
import torch
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset, TensorDataset
from torchvision import datasets, transforms
from timm.loss import LabelSmoothingCrossEntropy
import timm
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Helper class to ensure consistent label types
class TensorLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)


# Helper function to get class names
def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes


# Split training dataset into labeled and unlabeled subsets
def split_labeled_unlabeled(dataset, split_percentage=40):
    dataset_size = len(dataset)
    labeled_size = int(split_percentage / 100 * dataset_size)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    labeled_indices = indices[:labeled_size]
    unlabeled_indices = indices[labeled_size:]

    labeled_subset = Subset(dataset, labeled_indices)
    unlabeled_subset = Subset(dataset, unlabeled_indices)

    return labeled_subset, unlabeled_subset


# Generate pseudo-labels for unlabeled data
def generate_pseudo_labels(model, loader, threshold=0.9):
    print("Generating pseudo-labels...")
    model.eval()

    pseudo_data = []
    pseudo_labels = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)

            max_probs, preds = torch.max(probs, dim=1)
            mask = max_probs >= threshold

            pseudo_data.append(inputs[mask].cpu())
            pseudo_labels.append(preds[mask].cpu())

    if pseudo_data:
        pseudo_data = torch.cat(pseudo_data, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)
    else:
        pseudo_data = torch.empty(0, *inputs.shape[1:])
        pseudo_labels = torch.empty(0, dtype=torch.long)

    print(f"Generated {len(pseudo_labels)} pseudo-labels from unlabeled data.")
    return pseudo_data, pseudo_labels

def plot_loss(loss_per_iteration):
    # Plot loss over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(loss_per_iteration, label="Loss per iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Iterations")
    plt.legend()
    plt.grid()
    plt.show()

# Train a model
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=2):
    loss_per_iteration = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(dataloaders["train"], desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            loss_per_iteration.append(loss.item())
        epoch_loss = running_loss / len(dataloaders["train"].dataset)
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    plot_loss(loss_per_iteration)

# Train using pseudo-labeling
def train_pseudo_labeling(model, labeled_loader, unlabeled_loader, criterion, optimizer, scheduler, num_epochs=2, threshold=0.9):
    loss_per_iteration = []
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1} - Training on Labeled Data...")
        model.train()

        # Step 1: Train on labeled data
        running_loss = 0.0
        for inputs, labels in tqdm(labeled_loader, desc=f"Epoch {epoch+1}/Labeled Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        print(f"Epoch {epoch+1}, Labeled Data Loss: {running_loss / len(labeled_loader.dataset):.4f}")

        # Step 2: Generate pseudo-labels
        pseudo_data, pseudo_labels = generate_pseudo_labels(model, unlabeled_loader, threshold)

        if len(pseudo_labels) > 0:
            pseudo_dataset = TensorDataset(pseudo_data, pseudo_labels)
            labeled_dataset = TensorLabelsDataset(labeled_loader.dataset)

            combined_dataset = ConcatDataset([labeled_dataset, pseudo_dataset])
            combined_loader = DataLoader(combined_dataset, batch_size=labeled_loader.batch_size, shuffle=True)

            # Step 3: Train on combined data
            running_loss = 0.0
            for inputs, labels in tqdm(combined_loader, desc=f"Epoch {epoch+1}/Combined Training", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                loss_per_iteration.append(loss.item())

            print(f"Epoch {epoch+1}, Combined Data Loss: {running_loss / len(combined_loader.dataset):.4f}")

        torch.cuda.empty_cache()
    plot_loss(loss_per_iteration)



def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy


# Compare training paradigms
def compare_training_paradigms(model_fn, train_dataset, test_loader, val_loader, split_percentage=10, num_epochs=5):
    results = {}

    labeled_subset, unlabeled_subset = split_labeled_unlabeled(train_dataset, split_percentage)

    labeled_loader = DataLoader(labeled_subset, batch_size=32, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_subset, batch_size=32, shuffle=False)
    complete_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train on the complete dataset
    print("\nTraining on Complete Dataset...")
    model = model_fn()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    criterion = LabelSmoothingCrossEntropy()
    train_model(model, {"train": complete_loader, "val": val_loader}, criterion, optimizer, scheduler, num_epochs)
    results["Complete Dataset"] = evaluate_model(model, test_loader)

    # Train with pseudo-labeling
    print("\nTraining with Pseudo-Labeling...")
    model = model_fn()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    train_pseudo_labeling(model, labeled_loader, unlabeled_loader, criterion, optimizer, scheduler, num_epochs, threshold=0.01)
    results["Pseudo-Labeling"] = evaluate_model(model, test_loader)

    # Train on only labeled data
    print("\nTraining on Labeled Data Only...")
    model = model_fn()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    train_model(model, {"train": labeled_loader, "val": val_loader}, criterion, optimizer, scheduler, num_epochs)
    results["Labeled Data Only"] = evaluate_model(model, test_loader)

    return results


# Main script
if __name__ == '__main__':
    dataset_path = "CUB_200_2011/CUB_200_2011/images"

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter()], p=0.1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = datasets.ImageFolder(dataset_path, transform=transform_train)
    test_loader = DataLoader(datasets.ImageFolder(dataset_path, transform=transform_test_val), batch_size=32, shuffle=False)
    val_loader = DataLoader(datasets.ImageFolder(dataset_path, transform=transform_test_val), batch_size=32, shuffle=False)

    def model_fn():
        return timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(get_classes(dataset_path))).to(device)

    results = compare_training_paradigms(model_fn, train_dataset, test_loader, val_loader, split_percentage=50, num_epochs=5)

    print("\nResults:")
    for paradigm, accuracy in results.items():
        print(f"{paradigm}: {accuracy * 100:.2f}%")
