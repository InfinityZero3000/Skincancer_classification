"""
Gradio Web Interface for Skin Cancer Classification
Model: HybridViT (CNN + Vision Transformer)
"""

import os
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import numpy as np
import timm
from torchvision import transforms

# ========================== MODEL ARCHITECTURE ==========================
class CNNExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=128, patch=2, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class HybridViT(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.cnn = CNNExtractor()
        self.patch_embed = PatchEmbed()
        self.vit = timm.models.vision_transformer.vit_base_patch16_224(pretrained=False)
        self.vit.patch_embed = None
        self.classifier = nn.Linear(self.vit.embed_dim, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        logits = self.classifier(x[:, 0])
        return logits


# ========================== CONFIGURATION ==========================
CHECKPOINT_PATH = "best_model.pt"  # Đường dẫn đến model của bạn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 9

CLASS_NAMES = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Dermatofibroma',
    'Melanoma',
    'Nevus',
    'Pigmented Benign Keratosis',
    'Seborrheic Keratosis',
    'Squamous Cell Carcinoma',
    'Vascular Lesion'
]

# Class descriptions
CLASS_INFO = {
    'Actinic Keratosis': 'Precancerous skin lesion caused by sun exposure',
    'Basal Cell Carcinoma': 'Most common type of skin cancer, slow-growing',
    'Dermatofibroma': 'Benign skin growth, usually harmless',
    'Melanoma': 'Most dangerous form of skin cancer, requires immediate attention',
    'Nevus': 'Common mole, usually benign',
    'Pigmented Benign Keratosis': 'Non-cancerous brown spots',
    'Seborrheic Keratosis': 'Non-cancerous growth, common in older adults',
    'Squamous Cell Carcinoma': 'Second most common skin cancer',
    'Vascular Lesion': 'Blood vessel-related skin condition'
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ========================== LOAD MODEL ==========================
print(f"Loading model from {CHECKPOINT_PATH}...")
print(f"Using device: {DEVICE}")

model = HybridViT(num_classes=NUM_CLASSES).to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print("✅ Model loaded successfully!")
else:
    print(f"⚠️ Model file not found: {CHECKPOINT_PATH}")
    print("Please ensure 'best_model.pt' is in the same directory as this script.")

model.eval()


# ========================== PREDICTION FUNCTION ==========================
def predict(image):
    """
    Predict skin lesion type from image
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Dictionary with class probabilities and prediction info
    """
    try:
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            return {"error": "Invalid image format"}
        
        # Preprocess
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        # Get top prediction
        pred_idx = probabilities.argmax().item()
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probabilities[pred_idx].item()
        
        # Create results dictionary
        results = {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, probabilities.cpu().numpy())}
        
        # Sort by probability
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        # Additional info
        info = f"""
### 🔍 Prediction: **{pred_class}**
**Confidence:** {confidence*100:.2f}%

**Description:** {CLASS_INFO[pred_class]}

---
⚠️ **Disclaimer:** This is an AI prediction tool for educational purposes only. 
Always consult a qualified dermatologist for proper medical diagnosis.
"""
        
        return results, info
        
    except Exception as e:
        return {}, f"❌ Error: {str(e)}"


# ========================== GRADIO INTERFACE ==========================
# Custom CSS
custom_css = """
#title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: bold;
    margin-bottom: 0.5em;
}

#subtitle {
    text-align: center;
    color: #666;
    font-size: 1.2em;
    margin-bottom: 1em;
}

.gradio-container {
    max-width: 1200px !important;
}
"""

# Example images (you can add paths to sample images)
examples = [
    # Add paths to example images here
    # ["path/to/example1.jpg"],
    # ["path/to/example2.jpg"],
]

# Create interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 id='title'>🩺 Skin Cancer Classification</h1>")
    gr.Markdown("<p id='subtitle'>AI-powered Skin Lesion Analysis using Hybrid CNN-ViT Model</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input
            input_image = gr.Image(
                label="Upload Skin Lesion Image",
                type="pil",
                height=400
            )
            
            predict_btn = gr.Button("🔬 Analyze Image", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 📝 Instructions:
            1. Upload a clear image of the skin lesion
            2. Click "Analyze Image" to get prediction
            3. Review the classification results
            
            **Supported formats:** JPG, PNG, JPEG
            """)
            
        with gr.Column(scale=1):
            # Output
            output_label = gr.Label(
                label="Classification Results",
                num_top_classes=5
            )
            
            output_info = gr.Markdown(label="Detailed Information")
    
    # Add examples if available
    if examples:
        gr.Examples(
            examples=examples,
            inputs=input_image,
            label="Example Images"
        )
    
    # Model info
    gr.Markdown("""
    ---
    ### 🤖 Model Information
    - **Architecture:** Hybrid CNN + Vision Transformer (ViT)
    - **Dataset:** ISIC 2018 Skin Cancer Detection
    - **Classes:** 9 types of skin lesions
    - **Input Size:** 224x224 pixels
    
    ### ⚕️ Medical Disclaimer
    This tool is designed for **educational and research purposes only**. It should NOT be used as a substitute 
    for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider 
    for any skin concerns.
    
    ### 📊 Class Information
    The model can identify 9 types of skin lesions:
    - **Actinic Keratosis:** Precancerous lesion from sun damage
    - **Basal Cell Carcinoma:** Most common skin cancer
    - **Dermatofibroma:** Benign fibrous growth
    - **Melanoma:** Most dangerous skin cancer
    - **Nevus:** Common moles
    - **Pigmented Benign Keratosis:** Harmless brown spots
    - **Seborrheic Keratosis:** Age-related benign growths
    - **Squamous Cell Carcinoma:** Second most common skin cancer
    - **Vascular Lesion:** Blood vessel abnormalities
    """)
    
    # Connect prediction function
    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_label, output_info]
    )

# ========================== LAUNCH ==========================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Gradio Web Interface")
    print("="*60)
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Port number
        share=False,             # Set to True for public URL
        debug=True
    )
