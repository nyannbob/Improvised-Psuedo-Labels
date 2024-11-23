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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


# Threshold Scheduler Class
class ThresholdScheduler:
    def __init__(self, initial_threshold=0.3, max_threshold=0.9, min_threshold=0.6, increment=0.2):
        self.threshold = initial_threshold
        self.max_threshold = max_threshold
        self.increment = increment
        self.min_threshold = min_threshold

    def update(self, max_probs):
        avg_max_prob = max_probs.mean().item()
        if(avg_max_prob*2 >= self.max_threshold):
            factor = (self.threshold+avg_max_prob+self.max_threshold)/3
            self.increment = 0.05
            increment_f = self.increment
        else:
            factor = avg_max_prob*2

            if(factor<self.threshold):
                increment_f = self.increment/2
            else:
                self.increment = min(0.8,self.increment*2 )
                increment_f = self.increment
            
        print(self.increment)
        print(factor)
        self.threshold = min(self.max_threshold, self.threshold*(1-self.increment) + factor * increment_f)
        print(self.threshold)
        self.threshold = max(self.threshold,self.min_threshold)
        return self.threshold


# Generate pseudo-labels for unlabeled data
def generate_pseudo_labels(model, loader, threshold_scheduler):
    print("Generating pseudo-labels...")
    model.eval()

    pseudo_data = []
    pseudo_labels = []
    all_max_probs = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)

            max_probs, preds = torch.max(probs, dim=1)
            all_max_probs.append(max_probs)
            mask = max_probs >= threshold_scheduler.threshold

            pseudo_data.append(inputs[mask].cpu())
            pseudo_labels.append(preds[mask].cpu())

    if pseudo_data:
        pseudo_data = torch.cat(pseudo_data, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)
    else:
        pseudo_data = torch.empty(0, *inputs.shape[1:])
        pseudo_labels = torch.empty(0, dtype=torch.long)

    all_max_probs = torch.cat(all_max_probs, dim=0) if all_max_probs else torch.tensor([0.0])
    new_threshold = threshold_scheduler.update(all_max_probs)

    print(f"Generated {len(pseudo_labels)} pseudo-labels. New threshold: {new_threshold:.4f}")
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

# Train using pseudo-labeling
def train_pseudo_labeling(model, labeled_loader, unlabeled_loader, criterion, optimizer, scheduler, num_epochs=5, initial_threshold=0.4):
    threshold_scheduler = ThresholdScheduler(initial_threshold=initial_threshold)
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
        pseudo_data, pseudo_labels = generate_pseudo_labels(model, unlabeled_loader, threshold_scheduler)

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


# Evaluate model
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

    labeled_subset, unlabeled_subset = split_labeled_unlabeled(train_dataset, split_percentage=50)

    labeled_loader = DataLoader(labeled_subset, batch_size=32, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_subset, batch_size=32, shuffle=False)

    def model_fn():
        return timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(datasets.ImageFolder(dataset_path).classes)).to(device)

    print("\nTraining with Pseudo-Labeling and Threshold Scheduler...")
    model = model_fn()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    criterion = LabelSmoothingCrossEntropy()

    train_pseudo_labeling(model, labeled_loader, unlabeled_loader, criterion, optimizer, scheduler, num_epochs=10, initial_threshold=0.4)
    accuracy = evaluate_model(model, test_loader)

    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
