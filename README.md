# War Through the Lens of AI: Image Analysis of the Russia-Ukraine War in Spanish News

Computer Vision Project: Building an image Classification Model to detect war in News broadcasts. Subsequent inference with the obtained results is included. 

[Published Paper](https://repositori.upf.edu/items/5e3e3300-72b7-4e99-81bd-0811e6968caf) | [Dataset on Kaggle](https://www.kaggle.com/datasets/viktoriiayuzkiv/the-russia-ukraine-war-images-in-spanish-news)

## Overview

The representation of war in media significantly impacts public perception and political discourse. This study aims to analyse the evolution of visual reporting on the Russia-Ukraine war in Spanish news broadcasts. We investigate how the depiction of the war on three channels, Antena 3, La Sexta, and Telecinco, changed from December 2022 to April 2024, focusing on the evolution of war coverage and on-the-ground war imagery. 

To achieve this, we use a subset of over 10,000 manually labelled screenshots from news broadcasts covering the Russia-Ukraine war, distinguishing between war-related and non-war-related content. We use a pre-trained ResNet50 network to build a binary classification model capable of accurately classifying if an image is war-related. Using this model, we track how the imagery of the war evolved over time, finding that as the war progressed, the proportion of war-related imagery in news broadcasts decreased, as well as war coverage overall. This trend is consistent across all three channels. Furthermore, the fluctuations in war images do not strongly correlate with actual events and military actions, suggesting a divergence between media representation and reality.

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/War-Image-Classification.git
cd War-Image-Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set up Kaggle API Token

```bash
# Set your Kaggle API token (get it from https://www.kaggle.com/settings)
export KAGGLE_API_TOKEN=your_token_here
```

### 3. Download and Prepare Data

```bash
# Download dataset from Kaggle
python scripts/download_data.py

# Split into train/val/test sets
python scripts/prepare_data.py
```

### 4. Train the Model

```bash
# Train with default configuration
python scripts/train.py

# Or with a custom experiment name
python scripts/train.py --experiment my_experiment
```

### 5. Evaluate the Model

```bash
python scripts/evaluate.py --weights model_weights/my_experiment.pth
```

### 6. Run Inference

```bash
python scripts/predict.py \
    --weights model_weights/my_experiment.pth \
    --input path/to/images/ \
    --output predictions/output.csv
```

---

## Project Structure

```
War-Image-Classification/
├── configs/
│   └── config.yaml              # All hyperparameters and settings
├── data/
│   ├── raw/                     # Downloaded from Kaggle
│   │   ├── images/              # All 10,339 images
│   │   └── labels.csv           # Original labels
│   ├── processed/               # Train/val/test splits
│   │   ├── train_labels.csv
│   │   ├── val_labels.csv
│   │   └── test_labels.csv
│   └── external/                # Additional data (missiles, fatalities)
├── src/
│   ├── data/                    # Data loading modules
│   ├── models/                  # Model architectures
│   ├── training/                # Training loop
│   ├── evaluation/              # Metrics and explainability
│   ├── inference/               # Prediction and bias correction
│   └── utils/                   # Visualization utilities
├── scripts/
│   ├── download_data.py         # Download from Kaggle
│   ├── prepare_data.py          # Split data
│   ├── train.py                 # Train model
│   ├── evaluate.py              # Evaluate model
│   ├── predict.py               # Run inference
│   └── apply_bias_correction.py # Apply bias correction
├── model_weights/               # Saved model checkpoints
├── predictions/                 # Model predictions
├── notebooks/                   # Jupyter notebooks for EDA
└── report_charts/               # Generated figures
```

---

## Configuration

All settings are in `configs/config.yaml`. Key options:

### Model Architecture
```yaml
model:
  architecture: "resnet50"
  pretrained: true
  head:
    type: "custom"           # "base" or "custom"
    hidden_dims: [512, 256]  # Hidden layer sizes
  unfreeze_layers: ["layer3", "layer4", "fc"]  # Layers to fine-tune
```

### Training
```yaml
training:
  batch_size: 32
  epochs: 10
  patience: 3                # Early stopping patience
  optimizer:
    name: "adam"
    learning_rate: 0.0001
  use_weighted_sampler: true # Handle class imbalance
```

### Bias Correction
```yaml
bias_correction:
  enabled: true
  sensitivity: 0.84
  specificity: 0.97
  precision: 0.88
  npv: 0.96
```

---

## Scripts Reference

### `download_data.py`
Downloads the dataset from Kaggle.
```bash
python scripts/download_data.py [--config configs/config.yaml]
```

### `prepare_data.py`
Splits labels into train/val/test sets (70/15/15 by default).
```bash
python scripts/prepare_data.py [--config configs/config.yaml]
```

### `train.py`
Trains the classification model.
```bash
python scripts/train.py [--config configs/config.yaml] [--experiment NAME]
```

### `evaluate.py`
Evaluates a trained model on the test set.
```bash
python scripts/evaluate.py --weights model_weights/model.pth [--config configs/config.yaml]
```

### `predict.py`
Runs inference on unlabeled images.
```bash
python scripts/predict.py \
    --weights model_weights/model.pth \
    --input path/to/images/ \
    --output predictions/output.csv \
    [--apply-bias-correction]
```

### `apply_bias_correction.py`
Applies bias correction to existing predictions.
```bash
python scripts/apply_bias_correction.py \
    --input predictions/raw.csv \
    --output predictions/corrected.csv
```

---

## Methodology

### Data Collection
- Collected data from nightly news broadcasts of Antena 3, La Sexta, and Telecinco.
- Each video is approximately 30 minutes long, spanning from December 2022 to April 2024.
- Corresponding transcripts were also collected.

### Data Labeling
- Created a subset of over 10,000 images.
- Manually labeled images into three categories: Military, Physical Damage from the War, and Not War.
- Performed Human Level Performance test.
- The labelled dataset is published on Kaggle: [The Russia-Ukraine War Images on Spanish News](https://www.kaggle.com/datasets/viktoriiayuzkiv/the-russia-ukraine-war-images-in-spanish-news) 

### Model Training
- Experimented with various classification models.
- Selected a pre-trained ResNet50 model for its superior performance.
- Fine-tuned the model by adjusting hyperparameters and conducting error analysis.
- Best model achieves:
  - **Accuracy**: 94.27%
  - **Precision**: 88.06%
  - **Recall**: 84.00%
  - **F1 Score**: 85.98%

### Inferences
- Applied the trained model to generate predictions for the entire dataset.
- Performed a descriptive analysis of the visual representation trends.
- Compared trends with real events to assess alignment between news reporting and reality.

---

## Model Performance

| Model Variant | Accuracy | Precision | Recall | F1 |
|---------------|----------|-----------|--------|-----|
| Base (frozen) | 89.18% | 70.60% | 82.77% | 76.20% |
| Custom Head | 89.11% | 69.60% | 85.23% | 76.63% |
| Fine-tune 1 Layer | 90.53% | 72.12% | 89.23% | 79.78% |
| Fine-tune 2 Layers | 92.59% | 79.83% | 86.46% | 83.01% |
| Fine-tune 3 Layers | 92.20% | 77.13% | 89.23% | 82.74% |
| **Fine-tune All** | **92.78%** | **79.34%** | **88.64%** | **83.72%** |

---

## Dataset

The dataset contains 10,339 images from Spanish news broadcasts:

| Class | Count | Percentage |
|-------|-------|------------|
| Not War | 8,179 | 79% |
| Military | 1,298 | 13% |
| Physical Damage | 862 | 8% |

For binary classification, Military and Physical Damage are combined into "War" (21%).

---

## Notebooks

Jupyter notebooks are available for exploratory analysis:

- `notebooks/EDA.ipynb` - Exploratory data analysis
- `notebooks/ModelTraining.ipynb` - Interactive model training experiments
- `notebooks/ModelEvaluation.ipynb` - Detailed model evaluation
- `notebooks/BiasCorrection.ipynb` - Bias correction analysis
- `notebooks/PanelDataConversion.ipynb` - Panel data for regression analysis

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@thesis{war_image_classification_2024,
  title={War Through the Lens of AI: Image Analysis of the Russia-Ukraine War in Spanish News},
  author={Di Gianvito, Angelo and Gatland, Oliver and Yuzkiv, Viktoriia},
  year={2024},
  school={Universitat Pompeu Fabra},
  url={https://repositori.upf.edu/items/5e3e3300-72b7-4e99-81bd-0811e6968caf}
}
```

---

## Collaborators

- Angelo Di Gianvito
- Oliver Gatland
- Viktoriia Yuzkiv

---

## License

This project is licensed under the MIT License. The dataset is licensed under CC BY 4.0.
