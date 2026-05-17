# LA-DMDSC: Layered Attention Dynamic Margin Deep Simplex Classifier for Open-Set Medical Image Recognition
This repository contains the implementation code for the paper "LA-DMDSC: Layered Attention Dynamic Margin Deep Simplex Classifier for Open-Set Medical Image Recognition".

## Dataset Setup

### MedMNIST Dataset
The MedMNIST dataset can be downloaded from:
- **Official source**: https://zenodo.org/records/10519652

### Background Data
For extended experiments, download the 300K random images:
- **Source**: https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy

### Basic Training
Run the main training script with default parameters:
```bash
python NirvanaOSR.py --dataset dataset-name --dataroot ./data --outf ./results
```

### Hyperparameter Configuration

| Parameter | Description |
|-----------|-------------|
| `--batch-size` | Training batch size | 
| `--lr` | Learning rate | 
| `--max-epoch` | Maximum training epochs |
| `--optim` | Optimizer to be used |
| `--m-min` | Minimum margin for loss | 
| `--m-max` | Maximum margin for loss | 
| `--Expand` | Expand factor of centers |
| '--use-attn' | Self Attention |
| `--inter-weight` | Weight for inter-class loss |
| `--outlier-weight` | Weight for outlier triplet loss |
| `--model` | Backbone network to be used |
