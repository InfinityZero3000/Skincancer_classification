"""
Streamlit Web Interface for Skin Cancer Classification
Model: HybridViT (CNN + Vision Transformer)
"""

import os
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import timm
from torchvision import transforms
import plotly.graph_objects as go
import plotly.express as px

# ========================== PAGE CONFIG ==========================
st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
CHECKPOINT_PATH = r"D:\master\Cac van de hien dai TTNT\Project\Skincancer_VIT_Ver1.0_121125\best_model.pt"
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

CLASS_INFO = {
    'Actinic Keratosis': {
        'description': 'Precancerous skin lesion caused by prolonged sun exposure',
        'risk': 'Medium',
        'treatment': 'Can be treated with cryotherapy, topical medications, or photodynamic therapy',
        'color': '#FFA500'
    },
    'Basal Cell Carcinoma': {
        'description': 'Most common type of skin cancer, slow-growing and rarely spreads',
        'risk': 'Low-Medium',
        'treatment': 'Surgical removal, Mohs surgery, or radiation therapy',
        'color': '#FF6347'
    },
    'Dermatofibroma': {
        'description': 'Benign fibrous nodule, usually harmless',
        'risk': 'Low',
        'treatment': 'Usually no treatment needed, can be surgically removed if bothersome',
        'color': '#90EE90'
    },
    'Melanoma': {
        'description': 'Most dangerous form of skin cancer, can spread rapidly',
        'risk': 'High',
        'treatment': 'Requires immediate medical attention - surgery, immunotherapy, targeted therapy',
        'color': '#DC143C'
    },
    'Nevus': {
        'description': 'Common mole, usually benign',
        'risk': 'Very Low',
        'treatment': 'Monitor for changes, removal if suspicious',
        'color': '#87CEEB'
    },
    'Pigmented Benign Keratosis': {
        'description': 'Non-cancerous brown spots or growths',
        'risk': 'Very Low',
        'treatment': 'No treatment needed, cosmetic removal available',
        'color': '#98FB98'
    },
    'Seborrheic Keratosis': {
        'description': 'Non-cancerous growth common in older adults',
        'risk': 'Very Low',
        'treatment': 'No treatment needed, can be removed for cosmetic reasons',
        'color': '#DDA0DD'
    },
    'Squamous Cell Carcinoma': {
        'description': 'Second most common skin cancer, can spread if untreated',
        'risk': 'Medium',
        'treatment': 'Surgical removal, radiation therapy, or topical chemotherapy',
        'color': '#FF8C00'
    },
    'Vascular Lesion': {
        'description': 'Blood vessel-related skin condition',
        'risk': 'Low',
        'treatment': 'Laser therapy, surgical removal if needed',
        'color': '#FF69B4'
    }
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ========================== LOAD MODEL ==========================
@st.cache_resource
def load_model():
    """Load model with caching"""
    model = HybridViT(num_classes=NUM_CLASSES).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model.eval()
        return model, True
    else:
        return None, False


model, model_loaded = load_model()


# ========================== PREDICTION FUNCTION ==========================
def predict(image):
    """Predict skin lesion type"""
    if not model_loaded:
        return None, None, None
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Get results
    pred_idx = probabilities.argmax().item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probabilities[pred_idx].item()
    
    return pred_class, confidence, probabilities.cpu().numpy()


# ========================== UI COMPONENTS ==========================
def plot_probabilities(probs, class_names):
    """Create beautiful bar chart of probabilities with color coding"""
    df = pd.DataFrame({
        'Class': class_names,
        'Probability': probs * 100
    }).sort_values('Probability', ascending=True)
    
    # Add color based on probability
    colors = ['#FF4B4B' if p < 5 else '#FFA500' if p < 15 else '#4CAF50' if p > 30 else '#2196F3' 
              for p in df['Probability']]
    
    fig = go.Figure(data=[
        go.Bar(
            y=df['Class'],
            x=df['Probability'],
            orientation='h',
            text=[f'{p:.2f}%' for p in df['Probability']],
            textposition='outside',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            hovertemplate='<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '📊 Classification Probabilities',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1f77b4', 'family': 'Arial Black'}
        },
        xaxis_title="Probability (%)",
        yaxis_title="",
        height=450,
        font=dict(size=12),
        plot_bgcolor='rgba(240,240,240,0.5)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            range=[0, max(df['Probability']) * 1.15]
        ),
        margin=dict(l=20, r=100, t=60, b=40)
    )
    
    return fig


def plot_top5_pie(probs, class_names):
    """Create pie chart for top 5 predictions"""
    df = pd.DataFrame({
        'Class': class_names,
        'Probability': probs * 100
    }).sort_values('Probability', ascending=False).head(5)
    
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    fig = go.Figure(data=[
        go.Pie(
            labels=df['Class'],
            values=df['Probability'],
            hole=0.4,
            marker=dict(colors=colors_pie, line=dict(color='white', width=2)),
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>Probability: %{value:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '🥧 Top 5 Predictions',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1f77b4'}
        },
        height=400,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
    )
    
    return fig


def plot_probability_gauge(confidence):
    """Create gauge chart for confidence level"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size': 20}},
        delta={'reference': 70, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#FFE5E5'},
                {'range': [50, 70], 'color': '#FFF4E5'},
                {'range': [70, 85], 'color': '#E5F5E5'},
                {'range': [85, 100], 'color': '#E5FFE5'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def display_class_info(pred_class):
    """Display detailed information about predicted class"""
    info = CLASS_INFO[pred_class]
    
    st.markdown(f"### 📋 About {pred_class}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Description:**")
        st.info(info['description'])
        
    with col2:
        # Risk level with color coding
        risk_colors = {
            'Very Low': '#90EE90',
            'Low': '#87CEEB',
            'Low-Medium': '#FFA500',
            'Medium': '#FF8C00',
            'High': '#DC143C'
        }
        risk_color = risk_colors.get(info['risk'], '#808080')
        
        st.markdown(f"**Risk Level:**")
        st.markdown(
            f"<div style='background-color: {risk_color}; padding: 10px; "
            f"border-radius: 5px; text-align: center; color: white; font-weight: bold;'>"
            f"{info['risk']}</div>",
            unsafe_allow_html=True
        )
    
    st.markdown(f"**Recommended Treatment:**")
    st.success(info['treatment'])


# ========================== MAIN APP ==========================
def main():
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5em;
            font-weight: bold;
            margin-bottom: 0.2em;
        }
        .sub-header {
            text-align: center;
            color: #666;
            font-size: 1.3em;
            margin-bottom: 2em;
        }
        .stAlert {
            margin-top: 1em;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>🩺 Skin Cancer Classification</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-powered Skin Lesion Analysis using Hybrid CNN-ViT Model</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/hospital.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "ℹ️ About Model", "📊 Class Information"]
        )
        
        st.markdown("---")
        st.markdown("### System Info")
        st.info(f"**Device:** {DEVICE.upper()}")
        
        if model_loaded:
            st.success("✅ Model Loaded")
        else:
            st.error("❌ Model Not Found")
        
        st.markdown("---")
        st.markdown("### 📝 Quick Tips")
        st.markdown("""
        - Upload clear, well-lit images
        - Focus on the lesion area
        - Avoid blurry images
        - Consult a doctor for diagnosis
        """)
    
    # Main content
    if page == "🏠 Home":
        home_page()
    elif page == "ℹ️ About Model":
        about_page()
    elif page == "📊 Class Information":
        class_info_page()


def home_page():
    """Main prediction page"""
    if not model_loaded:
        st.error(f"⚠️ Model file not found: {CHECKPOINT_PATH}")
        st.info("Please ensure 'best_model.pt' is in the same directory as this script.")
        return
    
    st.markdown("### 📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose a skin lesion image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the skin lesion"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🖼️ Uploaded Image")
            st.image(image, use_container_width=True, caption="Input Image")
        
        with col2:
            st.markdown("#### 🎯 Prediction Result")
            
            with st.spinner("🔄 Analyzing image with AI..."):
                pred_class, confidence, probs = predict(image)
            
            if pred_class:
                # Display prediction with fancy styling
                info = CLASS_INFO[pred_class]
                
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, {info['color']}22 0%, {info['color']}44 100%); 
                                padding: 20px; border-radius: 10px; border-left: 5px solid {info['color']};
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                        <h2 style='color: {info['color']}; margin: 0;'>🎯 {pred_class}</h2>
                        <p style='font-size: 18px; margin: 10px 0 0 0; color: #333;'>
                            <strong>Confidence:</strong> <span style='color: {info['color']}; font-size: 24px; font-weight: bold;'>{confidence*100:.2f}%</span>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("")
                
                # Risk level badge
                risk_colors = {
                    'Very Low': ('#4CAF50', '🟢'),
                    'Low': ('#8BC34A', '🟢'),
                    'Low-Medium': ('#FFC107', '🟡'),
                    'Medium': ('#FF9800', '🟠'),
                    'High': ('#F44336', '🔴')
                }
                risk_color, risk_emoji = risk_colors.get(info['risk'], ('#808080', '⚪'))
                
                st.markdown(
                    f"""
                    <div style='background-color: {risk_color}; padding: 10px; border-radius: 5px; 
                                text-align: center; color: white; font-weight: bold; font-size: 16px;'>
                        {risk_emoji} Risk Level: {info['risk']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Confidence Gauge
        st.markdown("---")
        st.markdown("### 📈 Confidence Meter")
        col_gauge1, col_gauge2, col_gauge3 = st.columns([1, 2, 1])
        with col_gauge2:
            fig_gauge = plot_probability_gauge(confidence)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Probability visualizations
        st.markdown("---")
        st.markdown("### 📊 Probability Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📊 All Classes", "🥧 Top 5", "📋 Data Table"])
        
        with tab1:
            fig_bar = plot_probabilities(probs, CLASS_NAMES)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab2:
            col_pie1, col_pie2 = st.columns([3, 2])
            with col_pie1:
                fig_pie = plot_top5_pie(probs, CLASS_NAMES)
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_pie2:
                st.markdown("#### 🏆 Top 5 Predictions")
                df_top5 = pd.DataFrame({
                    'Rank': ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'],
                    'Class': [CLASS_NAMES[i] for i in np.argsort(probs)[::-1][:5]],
                    'Probability': [f"{probs[i]*100:.2f}%" for i in np.argsort(probs)[::-1][:5]]
                })
                st.dataframe(df_top5, use_container_width=True, hide_index=True)
        
        with tab3:
            st.markdown("#### 📋 Complete Probability Table")
            df_all = pd.DataFrame({
                'Class': CLASS_NAMES,
                'Probability (%)': [f"{p*100:.2f}" for p in probs],
                'Risk Level': [CLASS_INFO[c]['risk'] for c in CLASS_NAMES]
            }).sort_values('Probability (%)', ascending=False, key=lambda x: x.astype(float))
            st.dataframe(df_all, use_container_width=True, hide_index=True)
        
        # Show class information
        st.markdown("---")
        display_class_info(pred_class)
        
        # Medical disclaimer
        st.markdown("---")
        st.warning("""
        ⚕️ **Medical Disclaimer:** This tool is for educational purposes only and should NOT replace 
        professional medical advice. Always consult a qualified dermatologist for proper diagnosis and treatment.
        """)


def about_page():
    """Model information page"""
    st.markdown("## 🤖 About the Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        **Hybrid CNN + Vision Transformer (ViT)**
        
        The model combines:
        - **CNN Extractor:** 3 convolutional blocks for local feature extraction
        - **Patch Embedding:** Converts CNN features to patches
        - **Vision Transformer:** 12-layer transformer encoder
        - **Classifier:** Final linear layer for 9-class prediction
        """)
        
        st.markdown("### 📈 Training Details")
        st.markdown("""
        - **Dataset:** ISIC 2018 Skin Cancer Detection
        - **Classes:** 9 types of skin lesions
        - **Input Size:** 224×224 pixels
        - **Optimizer:** AdamW
        - **Loss Function:** Focal Loss with class weights
        - **Augmentation:** Rotation, flips, color jitter
        """)
    
    with col2:
        st.markdown("### 🎯 Performance Metrics")
        st.markdown("""
        The model was trained with:
        - Oversampling for class imbalance
        - Early stopping (patience=6)
        - Cosine annealing learning rate
        - Best model selection based on Macro F1-score
        """)
        
        st.markdown("### 📊 Model Statistics")
        st.info(f"""
        - **Parameters:** ~86M (ViT-Base backbone)
        - **Device:** {DEVICE.upper()}
        - **Framework:** PyTorch + TIMM
        """)


def class_info_page():
    """Display information about all classes"""
    st.markdown("## 📚 Skin Lesion Types")
    
    for class_name in CLASS_NAMES:
        with st.expander(f"📌 {class_name}"):
            info = CLASS_INFO[class_name]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Risk Level**")
                st.markdown(
                    f"<div style='background-color: {info['color']}; padding: 10px; "
                    f"border-radius: 5px; text-align: center; color: white; font-weight: bold;'>"
                    f"{info['risk']}</div>",
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown("**Description**")
                st.write(info['description'])
            
            with col3:
                st.markdown("**Treatment**")
                st.write(info['treatment'])


# ========================== RUN APP ==========================
if __name__ == "__main__":
    main()
