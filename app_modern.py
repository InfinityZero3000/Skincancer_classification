"""
Modern Streamlit Web Interface for Skin Cancer Classification
Model: HybridViT (CNN + Vision Transformer)
Version: 2.0 - Modern UI with Enhanced UX
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
    page_title="Skin Cancer AI Detection",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom icon library (using Unicode symbols only - no emojis)
ICONS = {
    'medical': '⚕',
    'camera': '◉',
    'chart': '▤',
    'search': '◈',
    'info': 'ℹ',
    'warning': '⚠',
    'check': '✓',
    'cross': '✗',
    'upload': '⬆',
    'download': '⬇',
    'gear': '⚙',
    'lightbulb': '◉',
    'trophy': '◆',
    'medal_1': '①',
    'medal_2': '②', 
    'medal_3': '③',
    'target': '◎',
    'clipboard': '▤',
    'pill': '●',
    'microscope': '◈',
    'dna': '◬',
    'brain': '◉',
    'clock': '◷',
    'globe': '◎',
    'book': '▣',
    'star': '★',
    'fire': '▲',
    'shield': '◆',
    'heart': '♥',
    'eye': '◉'
}

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
CHECKPOINT_PATH = "best_model.pt"
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

CLASS_NAMES_VI = [
    'Sừng hóa quang hóa',
    'Ung thư tế bào đáy',
    'U xơ da',
    'Ung thư hắc tố',
    'Nốt ruồi',
    'Sừng hóa lành tính có sắc tố',
    'Sừng hóa tiết nhờn',
    'Ung thư tế bào vảy',
    'Tổn thương mạch máu'
]

CLASS_INFO = {
    'Actinic Keratosis': {
        'name_vi': 'Sừng hóa quang hóa',
        'description': 'Tổn thương da tiền ung thư do tiếp xúc ánh nắng mặt trời kéo dài',
        'risk': 'Trung bình',
        'treatment': 'Có thể điều trị bằng đông lạnh, thuốc bôi tại chỗ hoặc liệu pháp quang động lực',
        'color': '#2196F3',
        'gradient': 'linear-gradient(135deg, #2196F3 0%, #1976D2 100%)'
    },
    'Basal Cell Carcinoma': {
        'name_vi': 'Ung thư tế bào đáy',
        'description': 'Loại ung thư da phổ biến nhất, phát triển chậm và hiếm khi lan rộng',
        'risk': 'Thấp-Trung bình',
        'treatment': 'Phẫu thuật cắt bỏ, phẫu thuật Mohs hoặc xạ trị',
        'color': '#E53935',
        'gradient': 'linear-gradient(135deg, #E53935 0%, #C62828 100%)'
    },
    'Dermatofibroma': {
        'name_vi': 'U xơ da',
        'description': 'Khối u xơ lành tính, thường vô hại',
        'risk': 'Thấp',
        'treatment': 'Thường không cần điều trị, có thể phẫu thuật nếu gây khó chịu',
        'color': '#1E88E5',
        'gradient': 'linear-gradient(135deg, #1E88E5 0%, #1565C0 100%)'
    },
    'Melanoma': {
        'name_vi': 'Ung thư hắc tố',
        'description': 'Dạng ung thư da nguy hiểm nhất, có thể lan nhanh',
        'risk': 'Cao',
        'treatment': 'Cần chú ý y tế ngay - phẫu thuật, liệu pháp miễn dịch, điều trị nhắm mục tiêu',
        'color': '#D32F2F',
        'gradient': 'linear-gradient(135deg, #D32F2F 0%, #B71C1C 100%)'
    },
    'Nevus': {
        'name_vi': 'Nốt ruồi',
        'description': 'Nốt ruồi thông thường, thường lành tính',
        'risk': 'Rất thấp',
        'treatment': 'Theo dõi các thay đổi, loại bỏ nếu nghi ngờ',
        'color': '#42A5F5',
        'gradient': 'linear-gradient(135deg, #42A5F5 0%, #1E88E5 100%)'
    },
    'Pigmented Benign Keratosis': {
        'name_vi': 'Sừng hóa lành tính có sắc tố',
        'description': 'Đốm hoặc mảng nâu không ung thư',
        'risk': 'Rất thấp',
        'treatment': 'Không cần điều trị, có thể loại bỏ vì mục đích thẩm mỹ',
        'color': '#0288D1',
        'gradient': 'linear-gradient(135deg, #0288D1 0%, #01579B 100%)'
    },
    'Seborrheic Keratosis': {
        'name_vi': 'Sừng hóa tiết nhờn',
        'description': 'U lành tính phổ biến ở người lớn tuổi',
        'risk': 'Rất thấp',
        'treatment': 'Không cần điều trị, có thể loại bỏ vì lý do thẩm mỹ',
        'color': '#1976D2',
        'gradient': 'linear-gradient(135deg, #1976D2 0%, #0D47A1 100%)'
    },
    'Squamous Cell Carcinoma': {
        'name_vi': 'Ung thư tế bào vảy',
        'description': 'Loại ung thư da phổ biến thứ hai, có thể lan rộng nếu không điều trị',
        'risk': 'Trung bình',
        'treatment': 'Phẫu thuật cắt bỏ, xạ trị hoặc hóa trị tại chỗ',
        'color': '#1565C0',
        'gradient': 'linear-gradient(135deg, #1565C0 0%, #0D47A1 100%)'
    },
    'Vascular Lesion': {
        'name_vi': 'Tổn thương mạch máu',
        'description': 'Tình trạng da liên quan đến mạch máu',
        'risk': 'Thấp',
        'treatment': 'Liệu pháp laser, phẫu thuật nếu cần',
        'color': '#039BE5',
        'gradient': 'linear-gradient(135deg, #039BE5 0%, #0277BD 100%)'
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
        return None, None, None, None
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    pred_idx = probabilities.argmax().item()
    pred_class = CLASS_NAMES[pred_idx]
    pred_class_vi = CLASS_NAMES_VI[pred_idx]
    confidence = probabilities[pred_idx].item()
    
    return pred_class, pred_class_vi, confidence, probabilities.cpu().numpy()


# ========================== MODERN UI COMPONENTS ==========================
def create_modern_metric(label, value, icon=""):
    """Create modern metric card"""
    return f"""
    <div style='
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 10px 0;
    '>
        <div style='color: #666; font-size: 14px; margin-bottom: 5px;'>{icon} {label}</div>
        <div style='color: #333; font-size: 24px; font-weight: 600;'>{value}</div>
    </div>
    """


def plot_modern_probabilities(probs, class_names_vi):
    """Create modern bar chart with blue theme"""
    df = pd.DataFrame({
        'Class': class_names_vi,
        'Probability': probs * 100
    }).sort_values('Probability', ascending=True)
    
    # Blue gradient colors
    colors = ['#E3F2FD' if p < 10 else '#90CAF9' if p < 30 else '#42A5F5' if p < 60 else '#1976D2' 
              for p in df['Probability']]
    
    fig = go.Figure(data=[
        go.Bar(
            y=df['Class'],
            x=df['Probability'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='#1565C0', width=1),
                cornerradius=10
            ),
            text=[f'{p:.1f}%' for p in df['Probability']],
            textposition='outside',
            textfont=dict(size=12, family='Inter', color='#1565C0'),
            hovertemplate='<b>%{y}</b><br>Xác suất: %{x:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '📊 Phân tích xác suất các loại bệnh',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1565C0', 'family': 'Inter'}
        },
        xaxis_title="Xác suất (%)",
        yaxis_title="",
        height=480,
        font=dict(size=13, family='Inter'),
        plot_bgcolor='rgba(227,242,253,0.3)',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(33,150,243,0.1)',
            range=[0, max(df['Probability']) * 1.15],
            tickfont=dict(color='#1565C0', size=12)
        ),
        yaxis=dict(
            tickfont=dict(size=12, color='#1565C0')
        ),
        margin=dict(l=20, r=120, t=70, b=50)
    )
    
    return fig


def plot_modern_gauge(confidence):
    """Create modern confidence gauge with blue theme"""
    # Blue gradient based on confidence
    color = '#E53935' if confidence < 0.5 else '#FB8C00' if confidence < 0.7 else '#1976D2'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "🎯 Độ tin cậy AI", 'font': {'size': 22, 'family': 'Inter', 'color': '#1565C0'}},
        number={'suffix': "%", 'font': {'size': 48, 'color': '#0D47A1'}},
        gauge={
            'axis': {
                'range': [None, 100], 
                'tickwidth': 2,
                'tickcolor': '#1976D2',
                'tickfont': {'size': 12, 'color': '#1565C0'}
            },
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': '#E3F2FD',
            'steps': [
                {'range': [0, 50], 'color': '#FFEBEE'},
                {'range': [50, 70], 'color': '#FFF3E0'},
                {'range': [70, 100], 'color': '#E3F2FD'}
            ],
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=40, t=70, b=30),
        paper_bgcolor='rgba(227,242,253,0.2)',
        font={'family': 'Inter'}
    )
    
    return fig


def plot_modern_donut(probs, class_names_vi):
    """Create modern donut chart with blue theme"""
    top5_idx = np.argsort(probs)[::-1][:5]
    top5_probs = probs[top5_idx]
    top5_names = [class_names_vi[i] for i in top5_idx]
    
    # Blue gradient palette
    colors = ['#0D47A1', '#1565C0', '#1976D2', '#1E88E5', '#42A5F5']
    
    fig = go.Figure(data=[
        go.Pie(
            labels=top5_names,
            values=top5_probs * 100,
            hole=0.6,
            marker=dict(
                colors=colors, 
                line=dict(color='white', width=3)
            ),
            textfont=dict(size=13, family='Inter', color='white'),
            textposition='inside',
            hovertemplate='<b>%{label}</b><br>%{value:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '🏆 Top 5 dự đoán',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Inter', 'color': '#1565C0'}
        },
        height=350,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=12, family='Inter', color='#1565C0'),
            bgcolor='rgba(227,242,253,0.3)',
            bordercolor='#1976D2',
            borderwidth=1
        ),
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor='rgba(227,242,253,0.2)'
    )
    
    return fig


# ========================== MODERN CSS ==========================
def load_modern_css():
    st.markdown("""
        <style>
        /* Import modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* Global styles */
        .main {
            background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
            font-family: 'Inter', sans-serif;
        }
        
        .block-container {
            padding: 2.5rem 2rem;
            background: linear-gradient(to bottom, #ffffff 0%, #f8fbff 100%);
            border-radius: 24px;
            margin: 20px;
            box-shadow: 0 20px 60px rgba(13,71,161,0.15);
        }
        
        /* Header styles */
        .modern-header {
            text-align: center;
            background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.2rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
            font-family: 'Inter', sans-serif;
            letter-spacing: -2px;
            text-shadow: 0 2px 10px rgba(25,118,210,0.1);
        }
        
        .modern-subtitle {
            text-align: center;
            color: #1565C0;
            font-size: 1.15rem;
            font-weight: 500;
            margin-bottom: 2.5rem;
            letter-spacing: 0.5px;
        }
        
        /* Card styles */
        .modern-card {
            background: white;
            border-radius: 20px;
            padding: 28px;
            box-shadow: 0 8px 24px rgba(25,118,210,0.08);
            border: 2px solid rgba(25,118,210,0.08);
            margin: 20px 0;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .modern-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 16px 40px rgba(25,118,210,0.12);
            border-color: rgba(25,118,210,0.15);
        }
        
        /* Button styles */
        .stButton>button {
            background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 14px 36px;
            font-weight: 700;
            font-size: 16px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 6px 20px rgba(25,118,210,0.35);
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(25,118,210,0.45);
            background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        }
        
        .stButton>button:active {
            transform: translateY(-1px);
        }
        
        /* File uploader */
        .uploadedFile {
            border-radius: 16px;
            border: 3px dashed #1976D2;
            padding: 24px;
            background: rgba(227,242,253,0.3);
            transition: all 0.3s;
        }
        
        .uploadedFile:hover {
            border-color: #0D47A1;
            background: rgba(227,242,253,0.5);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background: rgba(227,242,253,0.3);
            padding: 8px;
            border-radius: 14px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 14px 28px;
            font-weight: 600;
            color: #1565C0;
            background: transparent;
            transition: all 0.3s;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(25,118,210,0.1);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
            color: white !important;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1976D2 0%, #0D47A1 100%);
        }
        
        section[data-testid="stSidebar"] * {
            color: white !important;
        }
        
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stMarkdown {
            color: white !important;
        }
        
        /* Alert boxes */
        .stAlert {
            border-radius: 14px;
            border-left: 5px solid;
            padding: 16px 20px;
            backdrop-filter: blur(10px);
        }
        
        /* Info alert */
        div[data-baseweb="notification"][kind="info"] {
            background: rgba(227,242,253,0.9);
            border-left-color: #1976D2;
        }
        
        /* Success alert */
        div[data-baseweb="notification"][kind="success"] {
            background: rgba(232,245,233,0.9);
            border-left-color: #4CAF50;
        }
        
        /* Warning alert */
        div[data-baseweb="notification"][kind="warning"] {
            background: rgba(255,243,224,0.9);
            border-left-color: #FF9800;
        }
        
        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0D47A1;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 1rem;
            font-weight: 600;
            color: #1565C0;
        }
        
        /* Remove default padding */
        .element-container {
            margin-bottom: 1.2rem;
        }
        
        /* Image styling */
        img {
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        }
        
        /* Divider */
        hr {
            margin: 2rem 0;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #1976D2, transparent);
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #E3F2FD;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        }
        
        /* Code blocks */
        code {
            background: rgba(227,242,253,0.5);
            color: #0D47A1;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)


# ========================== MAIN APP ==========================
def main():
    load_modern_css()
    
    # Header with icon
    st.markdown("<h1 class='modern-header'>⚕ Hệ thống Phân tích Ung thư Da AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='modern-subtitle'>▤ Phân tích thông minh bằng mô hình Hybrid CNN-Vision Transformer</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙ Cài đặt hệ thống")
        
        language = st.selectbox(
            "◎ Ngôn ngữ / Language",
            ["Tiếng Việt", "English"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### ▤ Thông tin hệ thống")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**◈ Thiết bị:**")
            st.code(DEVICE.upper())
        with col2:
            st.markdown(f"**◉ Model:**")
            if model_loaded:
                st.success("✓ Sẵn sàng")
            else:
                st.error("✗ Lỗi")
        
        st.markdown("---")
        st.markdown("### ◉ Hướng dẫn sử dụng")
        st.info("""
        **① Bước 1:** Tải ảnh tổn thương da
        
        **② Bước 2:** Hệ thống tự động phân tích
        
        **③ Bước 3:** Xem kết quả chi tiết
        
        **⚕ Lưu ý:** Tham khảo bác sĩ để chẩn đoán chính xác
        """)
        
        st.markdown("---")
        st.markdown("### ▣ Về ứng dụng")
        st.markdown("""
        **◉ Phiên bản:** 2.0  
        **◈ Model:** HybridViT  
        **▤ Dataset:** ISIC 2018  
        **◎ Độ chính xác:** 85%+
        """)
    
    # Main content
    if not model_loaded:
        st.error(f"⚠ Không thể tải model từ: {CHECKPOINT_PATH}")
        st.info("Vui lòng đảm bảo file 'best_model.pt' có trong thư mục gốc.")
        return
    
    # File uploader section with improved layout
    col_upload1, col_upload2, col_upload3 = st.columns([1, 2, 1])
    with col_upload2:
        st.markdown("""
        <div style='
            text-align: center;
            background: linear-gradient(135deg, rgba(25,118,210,0.1) 0%, rgba(13,71,161,0.1) 100%);
            padding: 30px;
            border-radius: 20px;
            border: 2px solid rgba(25,118,210,0.2);
            margin: 20px 0;
        '>
            <h2 style='color: #1565C0; margin: 0 0 10px 0;'>📤 Tải ảnh lên để phân tích</h2>
            <p style='color: #1976D2; margin: 0;'>Hỗ trợ định dạng JPG, PNG, JPEG</p>
        </div>
        """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Chọn ảnh tổn thương da",
        type=['jpg', 'jpeg', 'png'],
        help="📷 Tải ảnh rõ nét của vùng da cần kiểm tra",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Image display and prediction
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div style='
                text-align: center;
                padding: 10px;
                background: linear-gradient(135deg, rgba(25,118,210,0.1) 0%, rgba(13,71,161,0.1) 100%);
                border-radius: 12px;
                margin-bottom: 15px;
            '>
                <h3 style='color: #1565C0; margin: 0;'>🖼 Ảnh gốc</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div style='
                text-align: center;
                padding: 10px;
                background: linear-gradient(135deg, rgba(25,118,210,0.1) 0%, rgba(13,71,161,0.1) 100%);
                border-radius: 12px;
                margin-bottom: 15px;
            '>
                <h3 style='color: #1565C0; margin: 0;'>🎯 Kết quả phân tích</h3>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("⏱ Đang phân tích bằng AI..."):
                pred_class, pred_class_vi, confidence, probs = predict(image)
            
            if pred_class:
                info = CLASS_INFO[pred_class]
                
                # Modern result card with blue theme
                st.markdown(
                    f"""
                    <div style='
                        background: {info['gradient']};
                        padding: 30px;
                        border-radius: 20px;
                        color: white;
                        box-shadow: 0 12px 32px rgba(25,118,210,0.3);
                        margin: 15px 0;
                        border: 3px solid rgba(255,255,255,0.3);
                    '>
                        <div style='text-align: center;'>
                            <h2 style='margin: 0 0 12px 0; font-size: 2rem; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.1);'>{pred_class_vi}</h2>
                            <p style='margin: 0 0 20px 0; opacity: 0.95; font-size: 0.95rem; font-weight: 500; letter-spacing: 0.5px;'>{pred_class}</p>
                            <div style='background: rgba(255,255,255,0.25); padding: 18px; border-radius: 12px; backdrop-filter: blur(10px);'>
                                <div style='font-size: 0.9rem; opacity: 0.95; margin-bottom: 8px; font-weight: 600; letter-spacing: 1px;'>🎯 ĐỘ TIN CẬY AI</div>
                                <div style='font-size: 2.8rem; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.1);'>{confidence*100:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Risk level badge with icons
                risk_info = {
                    'Rất thấp': ('#4CAF50', '✓', 'rgba(76,175,80,0.1)'),
                    'Thấp': ('#8BC34A', '✓', 'rgba(139,195,74,0.1)'),
                    'Thấp-Trung bình': ('#FFC107', '⚠', 'rgba(255,193,7,0.1)'),
                    'Trung bình': ('#FF9800', '⚠', 'rgba(255,152,0,0.1)'),
                    'Cao': ('#F44336', '⚠', 'rgba(244,67,54,0.1)')
                }
                risk_color, risk_icon, risk_bg = risk_info.get(info['risk'], ('#9E9E9E', '●', 'rgba(158,158,158,0.1)'))
                
                st.markdown(
                    f"""
                    <div style='
                        background: {risk_bg};
                        border: 3px solid {risk_color};
                        color: {risk_color};
                        padding: 16px 20px;
                        border-radius: 14px;
                        text-align: center;
                        font-weight: 700;
                        font-size: 1.1rem;
                        margin: 15px 0;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        letter-spacing: 0.5px;
                    '>
                        {risk_icon} MỨC ĐỘ NGUY HIỂM: {info['risk'].upper()}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Confidence gauge with improved section header
        st.markdown("---")
        st.markdown("""
        <div style='
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, rgba(25,118,210,0.08) 0%, rgba(13,71,161,0.08) 100%);
            border-radius: 16px;
            margin: 25px 0 15px 0;
            border: 2px solid rgba(25,118,210,0.15);
        '>
            <h2 style='color: #1565C0; margin: 0; font-size: 1.8rem; font-weight: 700;'>▤ Độ tin cậy của dự đoán</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
        with col_g2:
            fig_gauge = plot_modern_gauge(confidence)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Detailed analysis with improved section header
        st.markdown("---")
        st.markdown("""
        <div style='
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, rgba(25,118,210,0.08) 0%, rgba(13,71,161,0.08) 100%);
            border-radius: 16px;
            margin: 25px 0 20px 0;
            border: 2px solid rgba(25,118,210,0.15);
        '>
            <h2 style='color: #1565C0; margin: 0; font-size: 1.8rem; font-weight: 700;'>▤ Phân tích chi tiết</h2>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["▤ Tất cả các loại", "◆ Top 5", "▣ Thông tin bệnh"])
        
        with tab1:
            fig_bar = plot_modern_probabilities(probs, CLASS_NAMES_VI)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab2:
            col_t1, col_t2 = st.columns([3, 2])
            
            with col_t1:
                fig_donut = plot_modern_donut(probs, CLASS_NAMES_VI)
                st.plotly_chart(fig_donut, use_container_width=True)
            
            with col_t2:
                st.markdown("""
                <div style='
                    text-align: center;
                    padding: 12px;
                    background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
                    border-radius: 12px;
                    margin-bottom: 15px;
                '>
                    <h4 style='color: white; margin: 0; font-weight: 700; letter-spacing: 0.5px;'>◆ BẢNG XẾP HẠNG</h4>
                </div>
                """, unsafe_allow_html=True)
                
                top5_idx = np.argsort(probs)[::-1][:5]
                
                # Custom rank icons
                rank_icons = ['①', '②', '③', '④', '⑤']
                rank_colors = ['#0D47A1', '#1565C0', '#1976D2', '#1E88E5', '#42A5F5']
                
                for rank, idx in enumerate(top5_idx, 1):
                    st.markdown(
                        f"""
                        <div style='
                            background: linear-gradient(135deg, rgba(25,118,210,0.1) 0%, rgba(13,71,161,0.05) 100%);
                            border-radius: 12px;
                            padding: 14px 16px;
                            margin: 10px 0;
                            border-left: 5px solid {rank_colors[rank-1]};
                            box-shadow: 0 3px 8px rgba(25,118,210,0.15);
                            transition: all 0.3s;
                        '>
                            <div style='display: flex; align-items: center; justify-content: space-between;'>
                                <div style='display: flex; align-items: center; gap: 10px;'>
                                    <span style='
                                        background: {rank_colors[rank-1]};
                                        color: white;
                                        font-weight: 800;
                                        font-size: 1.3rem;
                                        width: 32px;
                                        height: 32px;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        border-radius: 50%;
                                    '>{rank}</span>
                                    <span style='font-weight: 600; color: #0D47A1; font-size: 0.95rem;'>{CLASS_NAMES_VI[idx]}</span>
                                </div>
                                <span style='
                                    font-weight: 800;
                                    color: {rank_colors[rank-1]};
                                    font-size: 1.1rem;
                                    background: white;
                                    padding: 4px 12px;
                                    border-radius: 8px;
                                '>{probs[idx]*100:.1f}%</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        with tab3:
            st.markdown(f"""
            <div style='
                text-align: center;
                padding: 16px;
                background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
                border-radius: 14px;
                margin-bottom: 20px;
            '>
                <h3 style='color: white; margin: 0; font-weight: 700; font-size: 1.4rem;'>📋 Thông tin về {pred_class_vi}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col_i1, col_i2 = st.columns(2)
            
            with col_i1:
                st.markdown("""
                <div style='
                    background: linear-gradient(135deg, rgba(33,150,243,0.08) 0%, rgba(25,118,210,0.05) 100%);
                    padding: 16px;
                    border-radius: 12px;
                    border-left: 4px solid #1976D2;
                    margin-bottom: 15px;
                '>
                    <h4 style='color: #1565C0; margin: 0 0 10px 0; font-weight: 700;'>📝 Mô tả chi tiết</h4>
                </div>
                """, unsafe_allow_html=True)
                st.info(info['description'])
                
            with col_i2:
                st.markdown("""
                <div style='
                    background: linear-gradient(135deg, rgba(33,150,243,0.08) 0%, rgba(25,118,210,0.05) 100%);
                    padding: 16px;
                    border-radius: 12px;
                    border-left: 4px solid #1976D2;
                    margin-bottom: 15px;
                '>
                    <h4 style='color: #1565C0; margin: 0 0 10px 0; font-weight: 700;'>💊 Phương pháp điều trị</h4>
                </div>
                """, unsafe_allow_html=True)
                st.success(info['treatment'])
            
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, rgba(25,118,210,0.1) 0%, rgba(13,71,161,0.05) 100%);
                padding: 18px;
                border-radius: 12px;
                border: 2px solid #1976D2;
                text-align: center;
                margin-top: 15px;
            '>
                <span style='color: #0D47A1; font-weight: 700; font-size: 1.1rem;'>⚠ Mức độ nguy hiểm:</span>
                <span style='
                    background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
                    color: white;
                    padding: 6px 20px;
                    border-radius: 8px;
                    margin-left: 10px;
                    font-weight: 800;
                    font-size: 1.1rem;
                '>{info['risk'].upper()}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Medical disclaimer with improved design
        st.markdown("---")
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, rgba(255,152,0,0.1) 0%, rgba(255,193,7,0.1) 100%);
            border: 3px solid #FF9800;
            border-radius: 16px;
            padding: 25px;
            margin: 25px 0;
            box-shadow: 0 6px 20px rgba(255,152,0,0.15);
        '>
            <h3 style='color: #E65100; margin: 0 0 15px 0; font-weight: 800; font-size: 1.3rem; text-align: center;'>
                ⚕ LƯU Ý Y TẾ QUAN TRỌNG
            </h3>
            <div style='color: #E65100; font-size: 1rem; line-height: 1.8;'>
                <p style='font-weight: 600; margin-bottom: 15px;'>
                    Ứng dụng này chỉ mang tính chất tham khảo và hỗ trợ, <strong>KHÔNG thay thế</strong> cho chẩn đoán y khoa chuyên nghiệp.
                </p>
                <div style='background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px; margin-top: 10px;'>
                    <p style='margin: 8px 0;'><strong>✓</strong> Luôn tham khảo ý kiến bác sĩ da liễu có chứng chỉ hành nghề</p>
                    <p style='margin: 8px 0;'><strong>✓</strong> Kết quả AI chỉ là công cụ hỗ trợ, không phải chẩn đoán cuối cùng</p>
                    <p style='margin: 8px 0;'><strong>✓</strong> Hãy khám sức khỏe định kỳ và theo dõi sự thay đổi của da</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ========================== RUN APP ==========================
if __name__ == "__main__":
    main()
