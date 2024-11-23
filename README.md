# Bird Species Classification using Semi-Supervised Learning with Dynamic Threshold Momentum

This repository implements a novel approach to semi-supervised learning for bird species classification using the CUB-200-2011 dataset. Our method enhances traditional pseudo-labeling by introducing a dynamic threshold momentum mechanism, achieving improved classification accuracy with limited labeled data.

## 🎯 Objectives

- Train a semi-supervised model using only 40% labeled data
- Generate and refine pseudo-labels for the remaining 60% unlabeled data
- Implement novel threshold momentum for pseudo-label selection
- Achieve competitive accuracy compared to fully supervised baselines
- Demonstrate real-world applicability on smartphone-captured bird images

## 📊 Dataset

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

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bird-classification-ssl.git
cd bird-classification-ssl

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- efficientnet-pytorch
- numpy
- pandas
- scikit-learn
- tqdm

## 🏗️ Model Architecture

We use EfficientNet-B0 as our base model due to its:
- Balanced accuracy-efficiency trade-off
- Suitability for fine-grained visual tasks
- Resource-efficient training on consumer GPUs
- Mobile-friendly architecture

## 🚀 Training Pipeline

### 1. Initial Supervised Training
```bash
python train_supervised.py --data_path ./data/CUB_200_2011 \
                         --labeled_fraction 0.4 \
                         --batch_size 64 \
                         --epochs 10
```

### 2. Pseudo-Label Generation with Threshold Momentum
```bash
python generate_pseudolabels.py --model_path ./checkpoints/supervised_model.pth \
                               --threshold_start 0.9 \
                               --momentum_factor 0.95
```

### 3. Semi-Supervised Training
```bash
python train_semisupervised.py --labeled_data ./data/labeled \
                              --pseudolabeled_data ./data/pseudolabeled \
                              --batch_size 64 \
                              --epochs 20
```

## 🔄 Novel Threshold Momentum Approach

Our approach introduces dynamic threshold scheduling using momentum:

1. Start with a high confidence threshold (e.g., 0.9)
2. Adjust threshold based on model's learning progress
3. Use momentum to smooth threshold changes
4. Automatically balance between precision and recall of pseudo-labels

```python
# Pseudo-code for threshold update
current_threshold = momentum_factor * previous_threshold + \
                   (1 - momentum_factor) * current_confidence
```

## 📈 Results

Model Performance Comparison:

| Model Type | Accuracy | Final Loss | Pseudo-Labeled Data |
|------------|----------|------------|-------------------|
| Full Supervised (100%) | 92.24% | 0.70 | N/A |
| Baseline (40% labeled) | 74.13% | 1.86 | N/A |
| Fixed Threshold | 86.20% | 0.93 | 838 |
| Threshold Momentum | 87.20% | 0.92 | 1817 |

## 👥 Contributors

- Nishant Kumar Singh (2104221)
- Vikrant Suresh Tripathi (2103141)
- Aditya Rajesh Bawangade (2103111)

## 📄 Citation

If you use this code in your research, please cite:
```bibtex
@misc{singh2024enhancing,
  title={Enhancing Bird Classification with Semi-Supervised Learning and Dynamic Threshold Momentum},
  author={Singh, Nishant Kumar and Tripathi, Vikrant Suresh and Bawangade, Aditya Rajesh},
  year={2024}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
