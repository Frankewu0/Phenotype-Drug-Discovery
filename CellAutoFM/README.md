# CellAutoFM: Automated Cell Image Analysis with Fine-tuned Vision Transformers

CellAutoFM is a deep learning pipeline for automated analysis of cellular autophagy images. The system uses fine-tuned Vision Transformers (ViT) with LoRA (Low-Rank Adaptation) to classify cellular conditions and generate image embeddings for unsupervised compound selection in drug screening applications.

## System Requirements

### Software Dependencies
- Python >= 3.12
- PyTorch >= 2.7.1
- torchvision >= 0.22.1
- transformers >= 4.53.0
- datasets >= 3.6.0
- peft == 0.13.2
- Pillow >= 11.3.0
- scikit-learn >= 1.7.0
- tqdm >= 4.67.1
- evaluate >= 0.4.4

### Operating Systems
Tested on:
- Linux (Ubuntu 20.04+)
- Windows 10/11
- macOS 10.15+

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
- **RAM**: Minimum 16GB, 32GB+ recommended for large datasets
- **Storage**: At least 10GB free space for models and data

## Installation Guide

### Instructions

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone the repository and navigate to the project directory**:
```bash
git clone <repository-url>
cd cellautofm
```

3. **Create virtual environment and install dependencies with uv**:
```bash
uv sync
```

4. **Activate the environment**:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Alternative Installation (Development)
For development or if you want to modify the code:
```bash
uv sync --dev
```

### Typical Install Time
- On a standard desktop computer: 3-7 minutes (uv is significantly faster than pip)
- On a high-performance workstation: 1-3 minutes

## Dataset Requirements

This project uses autophagy cell images with specific naming conventions:

### Training Dataset Structure
Training images should have filenames in the format:
```
prefix_timepoint_date-suffix_wellID_other.tif
```

For example: `sample_T1_220518-001_C_image.tif`

Where:
- `timepoint`: T1, T2, etc.
- `date`: 220518 (Fed wells: C,D,E,F,G,H; Starved wells: I,J,K,L,M,N) or 220524 (Fed wells: C,D,E; Starved wells: F,G,H,I,J,K,L,M)
- `wellID`: Well identifier (first character determines condition)

### Experimental Dataset Structure
For compound screening experiments, organize images as:
```
experimental_data_folder/
├── compound_id_1/
│   ├── compound_id_1_1.tif     # Red channel
│   ├── compound_id_1_2.tif     # Green channel
│   └── compound_id_1_Merge.tif # Merged channels
├── compound_id_2/
│   ├── compound_id_2_1.tif
│   ├── compound_id_2_2.tif
│   └── compound_id_2_Merge.tif
└── ...
```

## Demo

### Instructions to Run Demo

1. **Prepare your training dataset** following the naming convention described above

2. **Update configuration** in `main.py`:
```python
FOLDER_PATH = '/path/to/your/training/images/'  # Path to training dataset
EXPERIMENTAL_DATA_FOLDER = '/path/to/experimental/images/'  # Path to experimental images
```

3. **Run the complete pipeline**:
```bash
uv run python main.py
```

### Expected Output

The pipeline will execute in three stages:

#### 1. Data Preparation
- Parses filename metadata to extract timepoints and conditions (Fed vs Starved)
- Creates combined labels (e.g., "T1 Fed", "T1 Starved")
- Resizes images to 512×512 pixels
- Splits dataset into training (75%) and validation (25%) sets
- **Output**: Processed dataset saved to `./data/pretrain_autophagy_datasets_tif`

#### 2. Model Training
- Fine-tunes google/vit-large-patch16-224 with LoRA on autophagy classification
- Uses parameter-efficient fine-tuning to reduce computational requirements
- Trains for 20 epochs with early stopping based on validation accuracy
- **Output**: Trained model saved to `./models/vit-large-autophagy-lora`
- **Logs**: Training progress with accuracy metrics every 10 steps

#### 3. Image Representation Generation
- Processes experimental images through the fine-tuned model
- Generates normalized embeddings for similarity analysis
- **Output**: JSON files with embeddings:
  - `data/experiment_embedding_data_tif.json` (red channel images)
  - `data/experiment_embedding_data_tif2.json` (green channel images)  
  - `data/experiment_embedding_data_tif3.json` (merged channel images)

### Expected Run Time
- **Data preparation**: 5-20 minutes (depending on dataset size)
- **Model training**: 1-4 hours (depending on dataset size and hardware)
- **Embedding generation**: 10-30 minutes (depending on number of experimental images)
- **Total demo time on standard desktop**: 1.5-5 hours

## Instructions for Use

### How to Train Your Own Model

1. **Prepare training data** with proper filename structure as described above

2. **Customize training parameters** in `lora_training_ViT.py` or when calling `train_model()`:
```python
from model_training.lora_training_ViT import train_model

train_model(
    pretrain_model_name='google/vit-large-patch16-224',
    dataset_saved_path="./data/your_dataset",
    output_dir="./models/your_model",
    batch_size=32,  # Adjust based on GPU memory
    learning_rate=5e-4,
    num_train_epochs=20,
    cuda_device="0",  # GPU device number
    lora_r=16,
    lora_alpha=16,
    lora_dropout=0.1
)
```

### How to Generate Image Representations for Compound Screening

#### 1. Prepare Your Experimental Dataset
Organize compound images following the structure described in Dataset Requirements section.

#### 2. Generate Image Representations
```python
from unsupervised_compound_selection.image_representation_generation import unsupervised_image_representation_generation

results = unsupervised_image_representation_generation(
    pretrain_model_name='google/vit-large-patch16-224',
    model_name='./models/vit-large-autophagy-lora',
    experimental_data_folder='/path/to/experimental/images/'
)
```

#### 3. Compound Selection Analysis
Use the provided Jupyter notebook `unsupervised_compound_selection_based_on_representation.ipynb` to:
- Load generated embeddings from compound library images
- Combine embeddings from different channels (red, green, merged)
- Perform PCA analysis to visualize chemical space
- Identify compounds similar to reference autophagy inducers (FCCP) using cosine similarity
- Generate ranked lists of promising compounds for validation

### Running Individual Components

You can run individual components separately:

```bash
# Data preparation only
uv run python -c "
from data_preparation.training_data_preparation import train_data_preparation
train_data_preparation(
    '/path/to/training/images/',
    './data/autophagy_dataset',
    image_size=(512, 512),
    test_size=0.25
)"

# Training only (requires prepared dataset)
uv run python -c "
from model_training.lora_training_ViT import train_model
train_model(
    'google/vit-large-patch16-224',
    './data/autophagy_dataset',
    './models/autophagy_model'
)"

# Embedding generation only (requires trained model)
uv run python -c "
from unsupervised_compound_selection.image_representation_generation import unsupervised_image_representation_generation
unsupervised_image_representation_generation(
    'google/vit-large-patch16-224',
    './models/autophagy_model',
    '/path/to/experimental/images/'
)"
```

### Key Parameters

#### Training Parameters
- **`image_size`**: Input image dimensions (default: 512×512)
- **`batch_size`**: Training batch size (adjust based on GPU memory, default: 32)
- **`learning_rate`**: Learning rate for fine-tuning (default: 5e-4)
- **`num_train_epochs`**: Number of training epochs (default: 20)
- **`test_size`**: Fraction of data for validation (default: 0.25)

#### LoRA Parameters
- **`lora_r`**: LoRA rank parameter (default: 16, lower = fewer parameters)
- **`lora_alpha`**: LoRA alpha parameter (default: 16)
- **`lora_dropout`**: LoRA dropout rate (default: 0.1)

#### Hardware Parameters
- **`cuda_device`**: GPU device number (e.g., "0", "1", default: "3")
- **`fp16`**: Use mixed precision training (default: True)
- **`gradient_accumulation_steps`**: Steps to accumulate gradients (default: 4)

## Project Structure

```
cellautofm/
├── data_preparation/
│   └── training_data_preparation.py  # Dataset preprocessing
├── model_training/
│   └── lora_training_ViT.py          # LoRA fine-tuning
├── unsupervised_compound_selection/
│   ├── image_representation_generation.py  # Embedding generation
│   └── unsupervised_compound_selection_based_on_representation.ipynb  # Analysis notebook
├── main.py                           # Main pipeline script
├── pyproject.toml                    # Dependencies
└── README.md
```

## Troubleshooting

### Common Issues

#### Memory Issues
- **CUDA out of memory**: Reduce `batch_size` or increase `gradient_accumulation_steps`
- **System RAM exhausted**: Reduce `dataloader_num_workers` or process smaller batches

#### File Issues
- **File not found errors**: 
  - Verify training dataset paths and naming conventions
  - Check experimental data folder structure
  - Ensure image files are readable (.tif format)
- **Label generation errors**: Verify image filenames match expected format with proper timepoint and well IDs

#### Installation Issues
- **uv sync fails**: Try `uv sync --reinstall` or check Python version compatibility
- **CUDA not available**: Ensure PyTorch with CUDA support is installed
- **Import errors**: Verify all dependencies are installed with `uv sync`

### UV-Specific Tips
- **Check environment**: `uv run python --version` to verify Python version
- **Update dependencies**: `uv sync --upgrade` to update all packages
- **Add new dependencies**: `uv add <package-name>` to add packages to pyproject.toml
- **Troubleshoot environment**: `uv pip list` to see installed packages

### Model-Specific Tips
- **Low training accuracy**: Check if Fed/Starved labels are correctly assigned based on well IDs
- **Convergence issues**: Try adjusting learning rate or increasing warmup steps
- **Overfitting**: Increase dropout rate or reduce model complexity

### Expected Reproducibility
- Model accuracy should achieve >85% on Fed vs Starved classification on validation set
- Embedding similarities should be consistent across runs (±5% variation)
- PCA visualizations should maintain similar cluster patterns for reference compounds


## Support

For technical issues or questions:
- Open an issue in the repository
- Check the troubleshooting section above
- Contact the development team
