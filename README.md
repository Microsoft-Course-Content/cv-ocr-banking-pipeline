# 👁️ Computer Vision & OCR Pipeline for Banking

A production-grade **Computer Vision** and **OCR** system for banking document processing. Handles cheque reading, ID card verification, signature detection, fraud screening, and document quality assessment using **Azure AI Vision**, **OpenCV**, and **Azure OpenAI GPT-4o**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Azure](https://img.shields.io/badge/Azure-AI%20Vision%20%7C%20OpenAI-0078D4)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Image Upload   │────▶│  Quality Check   │────▶│  Document Type      │
│  (FastAPI)      │     │  (Blur, Rotation, │     │  Classification     │
│                 │     │   Resolution)     │     │  (GPT-4o Vision)    │
└─────────────────┘     └──────────────────┘     └─────────┬───────────┘
                                                            │
         ┌──────────────────────────────────────────────────┤
         ▼                    ▼                             ▼
┌─────────────────┐  ┌────────────────────┐  ┌──────────────────────────┐
│  Cheque Pipeline │  │  ID Card Pipeline  │  │  Signature Verification  │
│  ─────────────── │  │  ──────────────── │  │  ─────────────────────── │
│  • MICR Extract  │  │  • Face Detection  │  │  • Region Detection      │
│  • Amount OCR    │  │  • MRZ Reading     │  │  • Feature Extraction    │
│  • Date Extract  │  │  • Field Extract   │  │  • Similarity Scoring    │
│  • Bank Identify │  │  • Expiry Check    │  │  • Forgery Detection     │
│  • Fraud Flags   │  │  • Liveness Hints  │  │                          │
└────────┬────────┘  └────────┬───────────┘  └────────────┬─────────────┘
         │                    │                            │
         ▼                    ▼                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Structured Results + Confidence Scores             │
│              Fraud Flags • Compliance Status • Audit Trail            │
└──────────────────────────────────────────────────────────────────────┘
```

## ✨ Features

- **Cheque Processing**: MICR code extraction, amount reading (figures + words), payee detection, date extraction, bank identification
- **ID Card Verification**: Face detection, MRZ (Machine Readable Zone) parsing, field extraction, expiry validation
- **Signature Verification**: Region detection, feature extraction (ORB/SIFT), similarity scoring against reference signatures
- **Document Quality Assessment**: Blur detection (Laplacian), skew measurement, resolution check, noise estimation
- **Fraud Detection**: Tamper detection, copy-move forgery analysis, metadata inconsistency checks
- **Image Preprocessing**: Auto-deskew, contrast enhancement, noise reduction, border removal, DPI normalization
- **OCR Pipeline**: Azure AI Vision Read API + custom post-processing for banking-specific patterns

## 📁 Project Structure

```
project3-cv-ocr-banking/
├── src/
│   ├── main.py                        # FastAPI application + Web UI serving
│   ├── config.py                      # Configuration
│   ├── services/
│   │   ├── quality_assessor.py        # Image quality assessment
│   │   ├── ocr_engine.py             # Azure AI Vision OCR wrapper
│   │   ├── cheque_reader.py          # Cheque processing pipeline
│   │   ├── id_card_reader.py         # ID card verification + MRZ
│   │   ├── signature_verifier.py     # Signature detection & matching
│   │   ├── fraud_detector.py         # Fraud/tamper detection (ELA, copy-move)
│   │   └── blob_storage.py           # Azure Blob Storage connector
│   ├── models/
│   │   └── schemas.py                 # Pydantic models
│   └── utils/
│       └── cv_utils.py                # OpenCV utility functions
├── static/
│   └── index.html                     # Web UI — tabbed interface for all pipelines
├── data/sample_images/
├── tests/
│   └── test_cv_pipeline.py            # Quality + fraud detection tests
├── outputs/                           # Local results storage
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/cv-ocr-banking.git
cd cv-ocr-banking
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Edit with your Azure credentials
uvicorn src.main:app --reload --port 8002
```

### Open the Web UI

Open `http://localhost:8002` in your browser — a tabbed interface for all 6 CV/OCR tools loads automatically.

### Usage (CLI)

```bash
# Read a cheque
curl -X POST "http://localhost:8002/api/v1/cheque/read" -F "file=@cheque.png"

# Verify an ID card
curl -X POST "http://localhost:8002/api/v1/id-card/verify" -F "file=@passport.jpg"

# Detect fraud
curl -X POST "http://localhost:8002/api/v1/fraud/detect" -F "file=@suspicious_doc.png"
```

## ☁️ Azure Deployment (Web App)

```bash
# 1. Create resources
az group create --name rg-cv-ocr-banking --location uaenorth
az appservice plan create --name plan-cv-ocr --resource-group rg-cv-ocr-banking --sku B1 --is-linux
az webapp create --name cv-ocr-banking-app --resource-group rg-cv-ocr-banking \
  --plan plan-cv-ocr --runtime "PYTHON:3.11"

# 2. Configure environment
az webapp config appsettings set --name cv-ocr-banking-app --resource-group rg-cv-ocr-banking --settings \
  AZURE_VISION_ENDPOINT="https://your-vision.cognitiveservices.azure.com/" \
  AZURE_VISION_API_KEY="your-key" \
  AZURE_OPENAI_ENDPOINT="https://your-openai.openai.azure.com/" \
  AZURE_OPENAI_API_KEY="your-key" \
  AZURE_STORAGE_CONNECTION_STRING="your-connection-string"

# 3. Deploy
zip -r deploy.zip . -x "venv/*" "__pycache__/*" ".env"
az webapp deploy --name cv-ocr-banking-app --resource-group rg-cv-ocr-banking --src-path deploy.zip --type zip

# 4. Set startup command
az webapp config set --name cv-ocr-banking-app --resource-group rg-cv-ocr-banking \
  --startup-file "uvicorn src.main:app --host 0.0.0.0 --port 8000"
```

Live at: `https://cv-ocr-banking-app.azurewebsites.net`

### Storage Modes

| Mode | Condition | Images Stored | Results Stored |
|------|-----------|---------------|----------------|
| **Azure Blob** | Connection string set | `cv-ocr-documents/images/` | `cv-ocr-documents/results/` |
| **Local** | No connection string | `uploads/` | `outputs/` |

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/cheque/read` | Process and extract cheque data |
| `POST` | `/api/v1/id-card/verify` | Verify and extract ID card data |
| `POST` | `/api/v1/signature/verify` | Compare signature against reference |
| `POST` | `/api/v1/quality/assess` | Assess image quality for processing |
| `POST` | `/api/v1/ocr/extract` | General OCR text extraction |
| `POST` | `/api/v1/fraud/detect` | Run fraud detection checks |

## 🛠️ Tech Stack

- **Python 3.10+**, **FastAPI**, **OpenCV 4.9+**, **Pillow**
- **Azure AI Vision** — Read API (OCR), Image Analysis, Face Detection
- **Azure OpenAI GPT-4o** — Document classification, complex field extraction
- **NumPy** — Image array operations
- **scikit-image** — Advanced image analysis (SSIM, feature matching)

## 📝 License

MIT License

## 👤 Author

**Jalal Ahmed Khan** — Senior AI Consultant | Microsoft Certified Trainer
