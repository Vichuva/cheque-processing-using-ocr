# Cheque Processing Using OCR

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-red?style=flat&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow?style=flat)
![ZenML](https://img.shields.io/badge/ZenML-MLOps-green?style=flat)
![Gradio](https://img.shields.io/badge/Gradio-Interface-orange?style=flat)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?style=flat&logo=mlflow&logoColor=white)
![Label Studio](https://img.shields.io/badge/Label%20Studio-Annotation-purple?style=flat)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

An automated cheque processing system built with Donut Transformer and ZenML for end-to-end MLOps pipeline management.

## Overview

This project implements an OCR-free cheque processing system that extracts and validates information from bank cheque images. The system uses the Donut (Document Understanding Transformer) model for information extraction and includes a complete MLOps pipeline for data annotation, model training, deployment, and inference.

### Key Features

- **Information Extraction**: Automatically extracts payee name, amounts (words & figures), bank name, and cheque date
- **Smart Validation**: 
  - Verifies legal and courtesy amounts match
  - Detects stale cheques (older than 3 months)
  - Spell-checks extracted text
- **MLOps Pipeline**: Complete workflow using ZenML for data processing, training, and deployment
- **Web Interface**: Gradio-based demo application for easy testing

## Project Structure

```
cheque-easy-main/
├── app.py                          # Gradio web interface
├── predict_cheque_parser.py        # Prediction and validation logic
├── run_train_deploy.py             # Training & deployment pipeline runner
├── run_label_process_data.py       # Data labeling pipeline runner
├── params.py                       # Configuration parameters
├── requirements.txt                # Python dependencies
│
├── pipelines/                      # ZenML pipeline definitions
│   └── cheque_parser/
│       ├── labelling.py           # Data annotation pipeline
│       ├── data_postprocess.py    # Data processing pipeline
│       ├── train_deploy.py        # Training & deployment pipeline
│       └── inference_pipeline.py  # Inference pipeline
│
├── steps/                          # ZenML pipeline steps
│   └── cheque_parser/
│       ├── labelling/             # Annotation steps
│       ├── data_postprocess/      # Data processing steps
│       ├── train_donut/           # Training steps
│       └── inference/             # Inference steps
│
├── utils/                          # Utility modules
│   ├── create_pt_dataset.py       # Dataset creation utilities
│   ├── donut_pl_module.py         # PyTorch Lightning module
│   └── donut_utils.py             # Helper functions
│
├── materializers/                  # Custom ZenML materializers
│   ├── config_materializer.py     # Config serialization
│   └── donut_processor_materializer.py
│
├── zenml_stacks/                   # ZenML stack setup scripts
│   ├── label_data_process_stack.sh
│   └── train_inference_stack.sh
│
└── examples/                       # Example cheque images
    └── cheque_parser/
```

## Technology Stack

- **Model**: Donut Transformer (OCR-free document understanding)
- **MLOps Framework**: ZenML
- **Training**: PyTorch Lightning
- **Experiment Tracking**: MLflow
- **Data Annotation**: Label Studio
- **Web Interface**: Gradio
- **Validation**: SymSpell, word2number

## Installation

### Prerequisites

- Python 3.7, 3.8, or 3.9
- CUDA-capable GPU (recommended for training)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vichuva/cheque-processing-using-ocr.git
   cd cheque-processing-using-ocr
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize ZenML**
   ```bash
   zenml init
   zenml up
   ```

## Usage

### Running the Web Demo

Launch the Gradio interface to test cheque parsing:

```bash
python app.py
```

The interface allows you to:
- Upload cheque images
- View extracted information (payee, amounts, date, bank)
- Check amount validation status
- Detect stale cheques

### Running MLOps Pipelines

#### 1. Data Processing Pipeline

Processes raw cheque images and labels into training-ready format:

```bash
python run_train_deploy.py --pipeline_type=data_process
```

#### 2. Training Pipeline

Trains the Donut model with automatic evaluation and deployment:

```bash
python run_train_deploy.py --pipeline_type=train
```

Features:
- Loads preprocessed dataset
- Fine-tunes Donut model
- Evaluates on test set
- Automatically deploys if accuracy > 80%
- Logs experiments to MLflow

#### 3. Inference Pipeline

Runs predictions using the deployed model:

```bash
python run_train_deploy.py --pipeline_type=inference
```

#### 4. Data Labeling Pipeline (Optional)

For custom dataset annotation:

```bash
# Create Label Studio project
python run_label_process_data.py --pipeline_type=label

# Start annotation interface
zenml annotator dataset annotate <dataset_name>

# Retrieve labeled data
python run_label_process_data.py --pipeline_type=get_labelled_data

# Process labeled data
python run_label_process_data.py --pipeline_type=data_process
```

## Configuration

### Model Parameters

Edit `params.py` to customize:

```python
class DonutTrainParams:
    pretrained_ckpt = "nielsr/donut-base"
    image_size = [960, 720]
    max_length = 768
    batch_size = 1
    max_epochs = 30
    lr = 3e-5
    accelerator = "gpu"
```

### Deployment Parameters

```python
class ModelSaveDeployParams:
    workers = 3
    min_accuracy = 0.8  # Minimum accuracy for deployment
    timeout = 60
```

### Data Parameters

```python
class DataParams:
    annotation_file_path = "../cheques_dataset/cheques_label_file.csv"
    cheques_dataset_path = '../cheques_dataset/cheque_images'
    train_data_path = "../hf_cheques_data/train"
    val_data_path = "../hf_cheques_data/val"
    test_data_path = "../hf_cheques_data/test"
```

## ZenML Stack Setup

### For Annotation (with Azure)

Set environment variables:
```bash
export ANNOT_STACK_NAME="annotation_stack"
export AZURE_KEY_VAULT="your-key-vault"
export STORAGE_ACCOUNT="your-storage-account"
export BUCKET_NAME="az://your-bucket"
export STORAGE_ACCOUNT_KEY="your-access-key"
export LABEL_STUDIO_API_KEY="your-label-studio-token"
export LABEL_DATA_STORAGE_BUCKET_NAME="az://label-data-bucket"
```

Run setup:
```bash
bash zenml_stacks/label_data_process_stack.sh
```

### For Training & Inference

Set environment variables:
```bash
export TRAIN_STACK_NAME="training_stack"
export MLFLOW_TRACKING_URI="your-mlflow-uri"
export MLFLOW_USERNAME="your-username"
export MLFLOW_PASSWORD="your-password"
```

Run setup:
```bash
bash zenml_stacks/train_inference_stack.sh
```

## Extracted Fields

The system extracts the following information from cheques:

| Field | Description |
|-------|-------------|
| **Payee Name** | Recipient of the cheque |
| **Amount in Words** | Legal amount (written text) |
| **Amount in Figures** | Courtesy amount (numeric) |
| **Bank Name** | Issuing bank |
| **Cheque Date** | Date on the cheque |

## Validation Features

### Amount Matching
- Spell-checks the legal amount using SymSpell
- Converts words to numbers using word2number
- Compares with courtesy amount
- Returns match status

### Stale Cheque Detection
- Calculates months between current date and cheque date
- Flags cheques older than 3 months
- Helps prevent processing of expired cheques

## Model Architecture

The project uses **Donut (Document Understanding Transformer)**:

- **Encoder**: Vision Transformer (ViT) for image processing
- **Decoder**: Transformer decoder for text generation
- **Task**: Information extraction with custom prompt `<parse-cheque>`
- **Advantage**: No separate OCR step required

## Development

### Adding New Fields

To extract additional fields (e.g., MICR code, account number):

1. Update `params.py`:
   ```python
   cheque_parser_labels = ["payee_name", "bank_name", "amt_in_words", 
                          "amt_in_figures", "cheque_date", "micr_code"]
   ```

2. Update annotation configuration in `run_label_process_data.py`

3. Retrain the model with updated labels

### Custom Datasets

To use your own dataset:

1. Prepare data in the format specified in `DataParams`
2. Update paths in `params.py`
3. Modify `import_clean_data` step if needed
4. Run data processing pipeline

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Areas for improvement:

- Support for more banks and cheque formats
- Additional field extraction (MICR, account number, signature)
- Multi-language support
- Improved validation logic
- Performance optimizations

---

**Built with ZenML for end-to-end MLOps**
