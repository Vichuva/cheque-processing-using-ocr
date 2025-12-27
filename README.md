# 🏦 Cheque Processing Using OCR

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-red?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow?style=for-the-badge)
![ZenML](https://img.shields.io/badge/ZenML-MLOps-green?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-Interface-orange?style=for-the-badge)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Label Studio](https://img.shields.io/badge/Label%20Studio-Annotation-9B59B6?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

**An automated cheque processing system built with Donut Transformer and ZenML**  
*End-to-end MLOps pipeline for OCR-free document understanding*

</div>

---

## 📋 Overview

This project implements an **OCR-free cheque processing system** that extracts and validates information from bank cheque images. Using the **Donut (Document Understanding Transformer)** model, it provides a complete MLOps pipeline for data annotation, model training, deployment, and inference.

### ✨ Key Features

<table>
<tr>
<td width="50%">

#### 🔍 Information Extraction
- ✅ Payee name
- ✅ Amount in words & figures
- ✅ Bank name
- ✅ Cheque date

</td>
<td width="50%">

#### ✔️ Smart Validation
- ✅ Legal & courtesy amount matching
- ✅ Stale cheque detection (>3 months)
- ✅ Spell-check extracted text
- ✅ Date validation

</td>
</tr>
<tr>
<td width="50%">

#### 🚀 MLOps Pipeline
- ✅ Data processing & annotation
- ✅ Model training & evaluation
- ✅ Automated deployment
- ✅ Inference pipeline

</td>
<td width="50%">

#### 🎨 Web Interface
- ✅ Gradio-based demo
- ✅ Real-time predictions
- ✅ Visual feedback
- ✅ Example images included

</td>
</tr>
</table>

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| 🤖 **Model** | Donut Transformer (OCR-free) |
| ⚙️ **MLOps** | ZenML |
| 🔥 **Training** | PyTorch Lightning |
| 📊 **Tracking** | MLflow |
| 🏷️ **Annotation** | Label Studio |
| 🎨 **Interface** | Gradio |
| ✅ **Validation** | SymSpell, word2number |

---

## 📁 Project Structure

```
cheque-easy-main/
├── 🎨 app.py                          # Gradio web interface
├── 🔮 predict_cheque_parser.py        # Prediction and validation logic
├── 🚀 run_train_deploy.py             # Training & deployment pipeline runner
├── 🏷️ run_label_process_data.py       # Data labeling pipeline runner
├── ⚙️ params.py                       # Configuration parameters
├── 📦 requirements.txt                # Python dependencies
│
├── 📂 pipelines/                      # ZenML pipeline definitions
│   └── cheque_parser/
│       ├── labelling.py              # Data annotation pipeline
│       ├── data_postprocess.py       # Data processing pipeline
│       ├── train_deploy.py           # Training & deployment pipeline
│       └── inference_pipeline.py     # Inference pipeline
│
├── 📂 steps/                          # ZenML pipeline steps
│   └── cheque_parser/
│       ├── labelling/                # Annotation steps
│       ├── data_postprocess/         # Data processing steps
│       ├── train_donut/              # Training steps
│       └── inference/                # Inference steps
│
├── 📂 utils/                          # Utility modules
│   ├── create_pt_dataset.py          # Dataset creation utilities
│   ├── donut_pl_module.py            # PyTorch Lightning module
│   └── donut_utils.py                # Helper functions
│
├── 📂 materializers/                  # Custom ZenML materializers
│   ├── config_materializer.py        # Config serialization
│   └── donut_processor_materializer.py
│
├── 📂 zenml_stacks/                   # ZenML stack setup scripts
│   ├── label_data_process_stack.sh
│   └── train_inference_stack.sh
│
└── 📂 examples/                       # Example cheque images
    └── cheque_parser/
```

---

## 🚀 Quick Start

### Prerequisites

- 🐍 Python 3.7, 3.8, or 3.9
- 🎮 CUDA-capable GPU (recommended for training)

### Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/Vichuva/cheque-processing-using-ocr.git
cd cheque-processing-using-ocr

# 2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Initialize ZenML
zenml init
zenml up
```

### 🎨 Launch Web Demo

```bash
python app.py
```

**Features:**
- 📤 Upload cheque images
- 📊 View extracted information
- ✅ Check validation status
- ⚠️ Detect stale cheques

---

## 💻 Usage

### 📊 MLOps Pipelines

#### 1️⃣ Data Processing Pipeline

Process raw cheque images and labels:

```bash
python run_train_deploy.py --pipeline_type=data_process
```

#### 2️⃣ Training Pipeline

Train the Donut model with auto-deployment:

```bash
python run_train_deploy.py --pipeline_type=train
```

**Pipeline Features:**
- 📥 Loads preprocessed dataset
- 🔥 Fine-tunes Donut model
- 📊 Evaluates on test set
- 🚀 Auto-deploys if accuracy > 80%
- 📈 Logs experiments to MLflow

#### 3️⃣ Inference Pipeline

Run predictions with deployed model:

```bash
python run_train_deploy.py --pipeline_type=inference
```

#### 4️⃣ Data Labeling Pipeline (Optional)

Annotate custom datasets:

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

---

## ⚙️ Configuration

### 🤖 Model Parameters

Edit `params.py`:

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

### 🚀 Deployment Parameters

```python
class ModelSaveDeployParams:
    workers = 3
    min_accuracy = 0.8  # Minimum accuracy for deployment
    timeout = 60
```

### 📂 Data Parameters

```python
class DataParams:
    annotation_file_path = "../cheques_dataset/cheques_label_file.csv"
    cheques_dataset_path = '../cheques_dataset/cheque_images'
    train_data_path = "../hf_cheques_data/train"
    val_data_path = "../hf_cheques_data/val"
    test_data_path = "../hf_cheques_data/test"
```

---

## 🔧 ZenML Stack Setup

### 🏷️ Annotation Stack (Azure)

```bash
# Set environment variables
export ANNOT_STACK_NAME="annotation_stack"
export AZURE_KEY_VAULT="your-key-vault"
export STORAGE_ACCOUNT="your-storage-account"
export BUCKET_NAME="az://your-bucket"
export STORAGE_ACCOUNT_KEY="your-access-key"
export LABEL_STUDIO_API_KEY="your-label-studio-token"
export LABEL_DATA_STORAGE_BUCKET_NAME="az://label-data-bucket"

# Run setup
bash zenml_stacks/label_data_process_stack.sh
```

### 🚀 Training & Inference Stack

```bash
# Set environment variables
export TRAIN_STACK_NAME="training_stack"
export MLFLOW_TRACKING_URI="your-mlflow-uri"
export MLFLOW_USERNAME="your-username"
export MLFLOW_PASSWORD="your-password"

# Run setup
bash zenml_stacks/train_inference_stack.sh
```

---

## 📊 Extracted Fields

| Field | Description | Example |
|-------|-------------|---------|
| 👤 **Payee Name** | Recipient of the cheque | John Doe |
| 📝 **Amount in Words** | Legal amount (text) | Five Thousand Only |
| 💰 **Amount in Figures** | Courtesy amount (numeric) | 5000 |
| 🏦 **Bank Name** | Issuing bank | ICICI Bank |
| 📅 **Cheque Date** | Date on cheque | 27/12/2025 |

---

## ✅ Validation Features

### 💰 Amount Matching

1. **Spell-check** legal amount using SymSpell
2. **Convert** words to numbers using word2number
3. **Compare** with courtesy amount
4. **Return** match status ✅/❌

### ⚠️ Stale Cheque Detection

1. **Calculate** months between current date and cheque date
2. **Flag** cheques older than 3 months
3. **Prevent** processing of expired cheques

---

## 🧠 Model Architecture

**Donut (Document Understanding Transformer)**

```
┌─────────────────────────────────────┐
│   Input: Cheque Image [960x720]    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Vision Transformer (ViT) Encoder   │
│  • Processes image patches          │
│  • Extracts visual features         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Transformer Decoder                │
│  • Generates structured text        │
│  • Task prompt: <parse-cheque>      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Output: Extracted Information      │
│  • Payee, Amounts, Date, Bank       │
└─────────────────────────────────────┘
```

**✨ Advantages:**
- ✅ No separate OCR step required
- ✅ End-to-end trainable
- ✅ Understands document structure
- ✅ Better accuracy on complex layouts

---

## 🔨 Development

### ➕ Adding New Fields

Extract additional fields (MICR code, account number):

```python
# 1. Update params.py
cheque_parser_labels = [
    "payee_name", "bank_name", "amt_in_words", 
    "amt_in_figures", "cheque_date", "micr_code"
]

# 2. Update annotation config in run_label_process_data.py

# 3. Retrain model with updated labels
```

### 📦 Custom Datasets

Use your own dataset:

1. Prepare data in `DataParams` format
2. Update paths in `params.py`
3. Modify `import_clean_data` step if needed
4. Run data processing pipeline

---

## 🤝 Contributing

Contributions are welcome! **Areas for improvement:**

- 🏦 Support for more banks and cheque formats
- 🔍 Additional field extraction (MICR, account number, signature)
- 🌍 Multi-language support
- ⚡ Improved validation logic
- 🚀 Performance optimizations

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Built with ❤️ using **ZenML** for end-to-end MLOps

</div>
