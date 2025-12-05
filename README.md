# Skin Cancer AI Detection System

AI-powered web application using Deep Learning to classify 9 types of skin lesions with high accuracy, based on **HybridViT** architecture (CNN + Vision Transformer).

---

## Workflow

```mermaid
graph LR
    A[Upload Image] --> B[Preprocess]
    B --> C[CNN Feature Extraction]
    C --> D[Vision Transformer]
    D --> E[CBAM Attention]
    E --> F[Classification]
    F --> G[Display Results]
```

---

## Features

### Classification of 9 Skin Lesion Types:
1. Actinic Keratosis
2. Basal Cell Carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented Benign Keratosis
7. Seborrheic Keratosis
8. Squamous Cell Carcinoma
9. Vascular Lesion

### Interface:
- Professional design with blue theme
- Interactive charts (Plotly)
- Confidence gauge chart
- Top 5 predictions with probabilities
- Detailed information for each disease type

---

## Model Architecture

### HybridViT Architecture

The model uses **Hybrid CNN + Vision Transformer** architecture:

```
Input (224x224x3)
    ↓
CNN Extractor (3 Conv Blocks)
    ↓
Vision Transformer Base (timm)
    ↓
CBAM Attention Module
    ↓
Classifier (9 classes)
```

**Key Components:**
- **CNN Extractor**: 3 convolution blocks for local feature extraction
- **ViT Base**: Pretrained Vision Transformer for global feature learning
- **CBAM**: Convolutional Block Attention Module to enhance important regions
- **Classifier**: Fully connected layers with Dropout for classification

**Specifications:**
- Input size: 224×224 pixels
- Parameters: ~86M
- Training dataset: ISIC 2018
- Accuracy: 85%+

---

## Dataset

**ISIC 2018** (International Skin Imaging Collaboration)

- Training: ~10,000 images
- Testing: ~2,000 images
- Classes: 9 types
- Format: JPG, PNG
- Resolution: 224×224

---

## System Requirements

- CPU: Intel Core i5+ (GPU recommended)
- RAM: 8GB+
- Disk: 5GB
- Python: 3.8 - 3.11

---

## Installation

```bash
# Clone repository
git clone https://github.com/InfinityZero3000/Skincancer_VIT_Ver1.0_121125.git
cd Skincancer_VIT_Ver1.0_121125

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Note:** Model `best_model.pt` will be automatically downloaded from Google Drive on first run.

---

## Usage

### Run the web application

```bash
# Method 1: Using virtual environment
source .venv/bin/activate
streamlit run app_professional.py --server.port=8502

# Method 2: Run directly with Python from venv
.venv/bin/python -m streamlit run app_professional.py --server.port=8502
```

### Access the application

Open browser and visit:
- **Local**: http://localhost:8502
- **Network**: http://192.168.x.x:8502

### User Guide

1. **Prepare image**: Capture/select clear image with good lighting
2. **Upload image**: Click "Browse files" and select image (JPG/PNG)
3. **View results**: System analyzes and displays:
   - Predicted lesion type
   - Confidence level (%)
   - Top 5 predictions
   - Detailed disease information
   - Treatment recommendations
4. **Consult doctor**: Always consult a medical professional

---

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 85%+ |
| Precision | 83-88% |
| Recall | 82-87% |
| F1-Score | 82-87% |

### Best Performance

- Melanoma: 92% accuracy
- Basal Cell Carcinoma: 88% accuracy
- Nevus: 87% accuracy

---

## Technologies

- PyTorch (Deep Learning)
- Streamlit (Web Framework)
- Vision Transformer (ViT-Base)
- CBAM Attention
- Plotly (Visualization)

---

## Medical Disclaimer

**IMPORTANT:**

This application is FOR REFERENCE ONLY and does not replace medical diagnosis. AI results are support tools, not final diagnoses.

**Always consult a certified dermatologist.**

**Seek immediate medical attention if:**
- Moles change in shape, color, or size
- Sores don't heal after 2-3 weeks
- Unusual bleeding, itching, or pain in skin areas

---

## Authors

- Nguyen Thi Hong Quyen (Model Development)
- Nguyen Huu Thang (Web Application)

---

## License

This project is developed for research and educational purposes.

---

## Acknowledgments

- ISIC 2018: High-quality dataset
- Google Research: Vision Transformer architecture
- timm library: Pretrained models
- Streamlit: Web framework

---

## Support

Issues or questions? Open an issue on [GitHub Issues](https://github.com/InfinityZero3000/Skincancer_VIT_Ver1.0_121125/issues)

---

Your health is the top priority. Always consult a doctor!
