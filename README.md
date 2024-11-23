# Bird Species Classification using Pseudo Labeling and Threshold Scheduling

This repository implements a novel approach on the top of Pseudo labeling called 'Threshold Scheduling' for bird species classification using the CUB-200-2011 dataset. Our method enhances traditional pseudo-labeling by introducing a dynamic threshold momentum mechanism, achieving improved classification accuracy with limited labeled data.

## Objectives

- Train a semi-supervised model using only 40% labeled data
- Generate and refine pseudo-labels for the remaining 60% unlabeled data
- Implement novel threshold momentum for pseudo-label selection
- Achieve an accuracy comparable to the fully supervised baselines

## Dataset

We use the Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset, which contains:
- 200 bird species categories
- 11,788 images total
- 15 part locations per image
- 312 binary attributes per image
- Bounding box annotations

### Dataset Setup

1. Download the dataset:
```bash
# Create a data directory
mkdir data
cd data

# Download the CUB-200-2011 dataset
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

# Extract the dataset
tar -xzf CUB_200_2011.tgz
```

2. Create train/validation/test splits:
```python
# We use a 70:15:15 split ratio
train_size = 0.70
val_size = 0.15
test_size = 0.15
```

## Installation

```bash
# Clone the repository
git clone https://github.com/nyannbob/Improvised-Psuedo-Labels.git
cd git clone Improvised-Psuedo-Labels

# Install requirements
pip install -r requirements.txt
```

## Model Architecture

We use EfficientNet-B0 as our base model due to its:
- Balanced accuracy-efficiency trade-off
- Suitability for fine-grained visual tasks
- Resource-efficient training on consumer GPUs
- Mobile-friendly architecture

## Training stages

1. Initial Supervised Training
2. Pseudo-Label Generation with Threshold Momentum
3. Semi-Supervised Training

## Novel Threshold Momentum Approach

Our approach introduces dynamic threshold scheduling using momentum:

1. Start with a high confidence threshold (e.g., 0.8)
2. Adjust threshold based on model's learning progress
3. Use momentum to smooth threshold changes
4. Automatically balance between precision and recall of pseudo-labels

```python
# Pseudo-code for threshold update
current_threshold = momentum_factor * previous_threshold + \
                   (1 - momentum_factor) * current_confidence
```


## Visual Results

Here are some visual results and insights from our project:

| **Pseudo-Label Distribution** | **Model Performance on real world data** |
|-------------------------------|-------------------------------|
| ![Pseudo-Labels](barchart.png) | ![Accuracy](image.png) |


## Contributors

- Nishant Kumar Singh 
- Aditya Rajesh Bawangade 
- Vikrant Suresh Tripathi (2103141)



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
