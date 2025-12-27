# 🏦 ChequeEasy: OCR-Free Cheque Processing with AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-red)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)
![ZenML](https://img.shields.io/badge/ZenML-MLOps-green)
![License](https://img.shields.io/badge/License-MIT-purple)

**An end-to-end MLOps solution for automated cheque processing using Donut Transformer**

[Demo](https://huggingface.co/spaces/shivi/ChequeEasy) • [Dataset](https://huggingface.co/datasets/shivi/cheques_sample_data) • [Blog Post](https://medium.com/@shivalikasingh95/chequeeasy-banking-with-transformers-f49fb05960d3)

</div>

---

## 📋 Overview

**ChequeEasy** is an intelligent cheque processing system that automates the extraction and validation of information from bank cheques. Built with state-of-the-art AI and MLOps best practices, it streamlines the cheque approval process for both bank officials and customers.

### 🎯 Key Highlights

- **OCR-Free Processing**: Uses Donut (Document Understanding Transformer) - no traditional OCR required
- **End-to-End MLOps**: Complete pipeline from data annotation to model deployment using ZenML
- **Automated Validation**: Checks for amount matching and stale cheque detection
- **Production Ready**: Includes experiment tracking, model registry, and deployment workflows

---

## ✨ Features

### 🔍 Information Extraction
- **Payee Name**: Automatically extracts the recipient's name
- **Amount in Words**: Captures the legal amount written in text
- **Amount in Figures**: Extracts the courtesy amount (numeric)
- **Cheque Date**: Identifies the date on the cheque
- **Bank Name**: Recognizes the issuing bank

### ✅ Smart Validation
- **Amount Matching**: Verifies that legal and courtesy amounts match
- **Stale Cheque Detection**: Identifies cheques older than 3 months
- **Format Validation**: Ensures data integrity

### 🚀 MLOps Pipeline
- **Data Annotation**: Integrated Label Studio workflow
- **Model Training**: Automated fine-tuning with PyTorch Lightning
- **Experiment Tracking**: MLflow integration for versioning
- **Model Deployment**: Automated deployment based on performance metrics
- **Inference Pipeline**: Production-ready prediction service

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Model** | [Donut Transformer](https://arxiv.org/abs/2111.15664) (OCR-free VDU) |
| **MLOps Framework** | [ZenML](https://zenml.io/) |
| **Training** | PyTorch Lightning |
| **Experiment Tracking** | MLflow |
| **Data Annotation** | Label Studio |
| **Model Hub** | Hugging Face Transformers & Datasets |
| **Demo Interface** | Gradio |
| **Cloud Storage** | Azure Blob Storage (configurable) |

---

## 📁 Project Structure

```
cheque-easy-main/
├── app.py                          # Gradio demo application
├── predict_cheque_parser.py        # Inference script
├── run_label_process_data.py       # Labeling pipeline runner
├── run_train_deploy.py             # Training & deployment pipeline runner
├── params.py                       # Configuration parameters
├── requirements.txt                # Python dependencies
│
├── pipelines/                      # ZenML pipelines
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
│   └── donut_processor_materializer.py  # Processor serialization
│
└── zenml_stacks/                   # ZenML stack configurations
    ├── label_data_process_stack.sh
    └── train_inference_stack.sh
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7, 3.8, or 3.9
- CUDA-capable GPU (recommended for training)
- Azure account (optional, for cloud artifact storage)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vichuva/cheque-processing-using-ocr.git
   cd cheque-processing-using-ocr
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install ZenML (custom fork with Label Studio OCR support)**
   ```bash
   pip install git+https://github.com/shivalikasingh95/zenml.git@label_studio_ocr_config
   pip install "zenml[server]"
   ```

4. **Install Transformers (custom fork with fixes)**
   ```bash
   pip install git+https://github.com/shivalikasingh95/transformers.git@image_utils_fix
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Install additional dependencies**
   ```bash
   # For demo app
   pip install word2number gradio symspellpy
   
   # For MySQL backend (optional)
   sudo apt-get update
   sudo apt-get install python3-dev default-libmysqlclient-dev build-essential
   ```

7. **Initialize ZenML**
   ```bash
   zenml init
   zenml up
   ```

### Running the Demo

```bash
python app.py
```

This launches a Gradio interface where you can upload cheque images and see the extracted information.

---

## 📊 Dataset

The model is trained on a curated subset of the [Kaggle Cheque Images Dataset](https://www.kaggle.com/datasets/medali1992/cheque-images), focusing on 4 major Indian banks:
- Axis Bank
- Canara Bank
- HSBC
- ICICI Bank

**Download the prepared dataset:**
- 🤗 Hugging Face: [shivi/cheques_sample_data](https://huggingface.co/datasets/shivi/cheques_sample_data)

---

## 🔧 Usage

### 1. Data Processing Pipeline

Converts raw cheque images and labels into Hugging Face dataset format:

```bash
python run_train_deploy.py --pipeline_type=data_process
```

### 2. Training Pipeline

Fine-tunes the Donut model on the prepared dataset:

```bash
python run_train_deploy.py --pipeline_type=train
```

**Features:**
- Automatic experiment tracking with MLflow
- Model evaluation on test set
- Conditional deployment based on accuracy threshold (>80%)

### 3. Inference Pipeline

Runs predictions on new cheque images:

```bash
python run_train_deploy.py --pipeline_type=inference
```

### 4. Labeling Pipeline (Optional)

For custom dataset annotation using Label Studio:

```bash
# Create annotation project
python run_label_process_data.py --pipeline_type=label

# Start annotation
zenml annotator dataset annotate <dataset_name>

# Retrieve labeled data
python run_label_process_data.py --pipeline_type=get_labelled_data
```

---

## ⚙️ Configuration

### Environment Variables for Labeling Stack

```bash
export ANNOT_STACK_NAME="annotation_stack"
export AZURE_KEY_VAULT="your-key-vault"
export STORAGE_ACCOUNT="your-storage-account"
export BUCKET_NAME="az://your-bucket"
export STORAGE_ACCOUNT_KEY="your-access-key"
export LABEL_STUDIO_API_KEY="your-label-studio-token"
export LABEL_DATA_STORAGE_BUCKET_NAME="az://label-data-bucket"
```

### Environment Variables for Training Stack

```bash
export TRAIN_STACK_NAME="training_stack"
export MLFLOW_TRACKING_URI="your-mlflow-uri"
export MLFLOW_USERNAME="your-username"
export MLFLOW_PASSWORD="your-password"
```

### Model Parameters

Edit `params.py` to customize:
- Image size: `[960, 720]`
- Batch size: `1`
- Max epochs: `30`
- Learning rate: `3e-5`
- Minimum accuracy for deployment: `0.8`

---

## 🎯 Model Architecture

**Donut (Document Understanding Transformer)** is an OCR-free approach to Visual Document Understanding (VDU):

- **Encoder**: Vision Transformer (ViT) processes document images
- **Decoder**: Transformer decoder generates structured text output
- **No OCR Required**: End-to-end trainable without intermediate OCR steps
- **Task-Agnostic**: Can handle classification, extraction, and VQA

**Benefits over OCR-based approaches:**
- ✅ No need for separate OCR + downstream models
- ✅ Understands document structure natively
- ✅ No hand-crafted rules required
- ✅ Better handling of complex layouts

---

## 🎨 Demo

Try the live demo on Hugging Face Spaces:

🔗 **[ChequeEasy Demo](https://huggingface.co/spaces/shivi/ChequeEasy)**

Upload a cheque image and instantly see:
- Extracted information (payee, amounts, date, bank)
- Amount validation status
- Stale cheque warning

---

## 🏗️ ZenML Stack Setup

### Annotation Stack

```bash
bash zenml_stacks/label_data_process_stack.sh
```

**Components:**
- Artifact Store: Azure Blob Storage
- Secrets Manager: Azure Key Vault
- Annotator: Label Studio

### Training & Inference Stack

```bash
bash zenml_stacks/train_inference_stack.sh
```

**Components:**
- Experiment Tracker: MLflow
- Model Deployer: MLflow
- Artifact Store: Local or Cloud

---

## 📈 Performance

The model achieves:
- **Accuracy**: >80% on test set (deployment threshold)
- **Inference Speed**: Real-time processing on GPU
- **Supported Banks**: 4 major Indian banks (expandable)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Extend to more banks**: Add training data for additional banks
2. **Extract more fields**: MICR code, cheque number, account number
3. **Improve accuracy**: Fine-tune hyperparameters or augment data
4. **Add features**: Multi-language support, signature verification

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Original Dataset**: [Kaggle Cheque Images](https://www.kaggle.com/datasets/medali1992/cheque-images) by medali1992
- **Donut Model**: [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664) by Naver Clova
- **ZenML**: For the amazing MLOps framework
- **Hugging Face**: For Transformers and Datasets libraries

---

## 📚 References

- [Donut Paper](https://arxiv.org/abs/2111.15664) - Kim et al., 2021
- [ZenML Documentation](https://docs.zenml.io/)
- [Blog Post](https://medium.com/@shivalikasingh95/chequeeasy-banking-with-transformers-f49fb05960d3) - Detailed project walkthrough

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

<div align="center">

**Built with ❤️ for ZenML's Month of MLOps Competition**

⭐ Star this repo if you find it useful!

</div>
