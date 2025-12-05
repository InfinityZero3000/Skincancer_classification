"""
Giao diện Web Streamlit cho Phân loại Ung thư Da
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
    page_title="Phân loại Ung thư Da",
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
        'color': '#FFA500'
    },
    'Basal Cell Carcinoma': {
        'name_vi': 'Ung thư tế bào đáy',
        'description': 'Loại ung thư da phổ biến nhất, phát triển chậm và hiếm khi lan rộng',
        'risk': 'Thấp-Trung bình',
        'treatment': 'Phẫu thuật cắt bỏ, phẫu thuật Mohs hoặc xạ trị',
        'color': '#FF6347'
    },
    'Dermatofibroma': {
        'name_vi': 'U xơ da',
        'description': 'Khối u xơ lành tính, thường vô hại',
        'risk': 'Thấp',
        'treatment': 'Thường không cần điều trị, có thể phẫu thuật nếu gây khó chịu',
        'color': '#90EE90'
    },
    'Melanoma': {
        'name_vi': 'Ung thư hắc tố',
        'description': 'Dạng ung thư da nguy hiểm nhất, có thể lan nhanh',
        'risk': 'Cao',
        'treatment': 'Cần chú ý y tế ngay - phẫu thuật, liệu pháp miễn dịch, điều trị nhắm mục tiêu',
        'color': '#DC143C'
    },
    'Nevus': {
        'name_vi': 'Nốt ruồi',
        'description': 'Nốt ruồi thông thường, thường lành tính',
        'risk': 'Rất thấp',
        'treatment': 'Theo dõi các thay đổi, loại bỏ nếu nghi ngờ',
        'color': '#87CEEB'
    },
    'Pigmented Benign Keratosis': {
        'name_vi': 'Sừng hóa lành tính có sắc tố',
        'description': 'Đốm hoặc mảng nâu không ung thư',
        'risk': 'Rất thấp',
        'treatment': 'Không cần điều trị, có thể loại bỏ vì mục đích thẩm mỹ',
        'color': '#98FB98'
    },
    'Seborrheic Keratosis': {
        'name_vi': 'Sừng hóa tiết nhờn',
        'description': 'U lành tính phổ biến ở người lớn tuổi',
        'risk': 'Rất thấp',
        'treatment': 'Không cần điều trị, có thể loại bỏ vì lý do thẩm mỹ',
        'color': '#DDA0DD'
    },
    'Squamous Cell Carcinoma': {
        'name_vi': 'Ung thư tế bào vảy',
        'description': 'Loại ung thư da phổ biến thứ hai, có thể lan rộng nếu không điều trị',
        'risk': 'Trung bình',
        'treatment': 'Phẫu thuật cắt bỏ, xạ trị hoặc hóa trị tại chỗ',
        'color': '#FF8C00'
    },
    'Vascular Lesion': {
        'name_vi': 'Tổn thương mạch máu',
        'description': 'Tình trạng da liên quan đến mạch máu',
        'risk': 'Thấp',
        'treatment': 'Liệu pháp laser, phẫu thuật nếu cần',
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
        return None, None, None, None
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Get results
    pred_idx = probabilities.argmax().item()
    pred_class = CLASS_NAMES[pred_idx]
    pred_class_vi = CLASS_NAMES_VI[pred_idx]
    confidence = probabilities[pred_idx].item()
    
    return pred_class, pred_class_vi, confidence, probabilities.cpu().numpy()


# ========================== UI COMPONENTS ==========================
def plot_probabilities(probs, class_names_vi):
    """Create beautiful bar chart of probabilities"""
    df = pd.DataFrame({
        'Class': class_names_vi,
        'Probability': probs * 100
    }).sort_values('Probability', ascending=True)
    
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
            hovertemplate='<b>%{y}</b><br>Xác suất: %{x:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '📊 Xác Suất Phân Loại',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1f77b4', 'family': 'Arial Black'}
        },
        xaxis_title="Xác suất (%)",
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


def plot_top5_pie(probs, class_names_vi):
    """Create pie chart for top 5 predictions"""
    df = pd.DataFrame({
        'Class': class_names_vi,
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
            hovertemplate='<b>%{label}</b><br>Xác suất: %{value:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '🥧 Top 5 Dự Đoán',
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
        title={'text': "Độ Tin Cậy", 'font': {'size': 20}},
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


def display_class_info(pred_class, pred_class_vi):
    """Display detailed information about predicted class"""
    info = CLASS_INFO[pred_class]
    
    st.markdown(f"### 📋 Về {pred_class_vi}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Mô tả:**")
        st.info(info['description'])
        
    with col2:
        risk_colors = {
            'Rất thấp': '#90EE90',
            'Thấp': '#87CEEB',
            'Thấp-Trung bình': '#FFA500',
            'Trung bình': '#FF8C00',
            'Cao': '#DC143C'
        }
        risk_color = risk_colors.get(info['risk'], '#808080')
        
        st.markdown(f"**Mức độ nguy hiểm:**")
        st.markdown(
            f"<div style='background-color: {risk_color}; padding: 10px; "
            f"border-radius: 5px; text-align: center; color: white; font-weight: bold;'>"
            f"{info['risk']}</div>",
            unsafe_allow_html=True
        )
    
    st.markdown(f"**Phương pháp điều trị khuyến nghị:**")
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
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>🩺 Phân Loại Ung Thư Da</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Phân tích tổn thương da bằng AI sử dụng mô hình Hybrid CNN-ViT</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/hospital.png", width=80)
        st.title("Điều hướng")
        
        page = st.radio(
            "Chọn trang:",
            ["🏠 Trang chủ", "ℹ️ Về mô hình", "📊 Thông tin các lớp"]
        )
        
        st.markdown("---")
        st.markdown("### Thông tin hệ thống")
        st.info(f"**Thiết bị:** {DEVICE.upper()}")
        
        if model_loaded:
            st.success("✅ Đã tải mô hình")
        else:
            st.error("❌ Không tìm thấy mô hình")
        
        st.markdown("---")
        st.markdown("### 📝 Gợi ý sử dụng")
        st.markdown("""
        - Tải ảnh rõ nét, đủ ánh sáng
        - Tập trung vào vùng tổn thương
        - Tránh ảnh mờ, không rõ
        - Tham khảo bác sĩ để chẩn đoán
        """)
    
    # Main content
    if page == "🏠 Trang chủ":
        home_page()
    elif page == "ℹ️ Về mô hình":
        about_page()
    elif page == "📊 Thông tin các lớp":
        class_info_page()


def home_page():
    """Main prediction page"""
    if not model_loaded:
        st.error(f"⚠️ Không tìm thấy file mô hình: {CHECKPOINT_PATH}")
        st.info("Vui lòng đảm bảo file 'best_model.pt' nằm trong cùng thư mục với script này.")
        return
    
    st.markdown("### 📤 Tải ảnh lên")
    
    uploaded_file = st.file_uploader(
        "Chọn ảnh tổn thương da...",
        type=['jpg', 'jpeg', 'png'],
        help="Tải lên ảnh rõ nét của vùng tổn thương da"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🖼️ Ảnh đã tải lên")
            st.image(image, use_container_width=True, caption="Ảnh đầu vào")
        
        with col2:
            st.markdown("#### 🎯 Kết quả dự đoán")
            
            with st.spinner("🔄 Đang phân tích ảnh bằng AI..."):
                pred_class, pred_class_vi, confidence, probs = predict(image)
            
            if pred_class:
                info = CLASS_INFO[pred_class]
                
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, {info['color']}22 0%, {info['color']}44 100%); 
                                padding: 20px; border-radius: 10px; border-left: 5px solid {info['color']};
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                        <h2 style='color: {info['color']}; margin: 0;'>🎯 {pred_class_vi}</h2>
                        <p style='font-size: 14px; margin: 5px 0; color: #666;'>{pred_class}</p>
                        <p style='font-size: 18px; margin: 10px 0 0 0; color: #333;'>
                            <strong>Độ tin cậy:</strong> <span style='color: {info['color']}; font-size: 24px; font-weight: bold;'>{confidence*100:.2f}%</span>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("")
                
                risk_colors = {
                    'Rất thấp': ('#4CAF50', '🟢'),
                    'Thấp': ('#8BC34A', '🟢'),
                    'Thấp-Trung bình': ('#FFC107', '🟡'),
                    'Trung bình': ('#FF9800', '🟠'),
                    'Cao': ('#F44336', '🔴')
                }
                risk_color, risk_emoji = risk_colors.get(info['risk'], ('#808080', '⚪'))
                
                st.markdown(
                    f"""
                    <div style='background-color: {risk_color}; padding: 10px; border-radius: 5px; 
                                text-align: center; color: white; font-weight: bold; font-size: 16px;'>
                        {risk_emoji} Mức độ nguy hiểm: {info['risk']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Confidence Gauge
        st.markdown("---")
        st.markdown("### 📈 Đồng hồ độ tin cậy")
        col_gauge1, col_gauge2, col_gauge3 = st.columns([1, 2, 1])
        with col_gauge2:
            fig_gauge = plot_probability_gauge(confidence)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Probability visualizations
        st.markdown("---")
        st.markdown("### 📊 Phân tích xác suất")
        
        tab1, tab2, tab3 = st.tabs(["📊 Tất cả lớp", "🥧 Top 5", "📋 Bảng dữ liệu"])
        
        with tab1:
            fig_bar = plot_probabilities(probs, CLASS_NAMES_VI)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab2:
            col_pie1, col_pie2 = st.columns([3, 2])
            with col_pie1:
                fig_pie = plot_top5_pie(probs, CLASS_NAMES_VI)
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_pie2:
                st.markdown("#### 🏆 Top 5 dự đoán")
                df_top5 = pd.DataFrame({
                    'Hạng': ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'],
                    'Loại bệnh': [CLASS_NAMES_VI[i] for i in np.argsort(probs)[::-1][:5]],
                    'Xác suất': [f"{probs[i]*100:.2f}%" for i in np.argsort(probs)[::-1][:5]]
                })
                st.dataframe(df_top5, use_container_width=True, hide_index=True)
        
        with tab3:
            st.markdown("#### 📋 Bảng xác suất đầy đủ")
            df_all = pd.DataFrame({
                'Loại bệnh': CLASS_NAMES_VI,
                'Xác suất (%)': [f"{p*100:.2f}" for p in probs],
                'Mức độ nguy hiểm': [CLASS_INFO[c]['risk'] for c in CLASS_NAMES]
            }).sort_values('Xác suất (%)', ascending=False, key=lambda x: x.astype(float))
            st.dataframe(df_all, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        display_class_info(pred_class, pred_class_vi)
        
        st.markdown("---")
        st.warning("""
        ⚕️ **Tuyên bố Y tế:** Công cụ này chỉ dành cho mục đích giáo dục và KHÔNG thay thế 
        tư vấn y tế chuyên nghiệp. Luôn tham khảo bác sĩ da liễu có trình độ để được chẩn đoán và điều trị đúng cách.
        """)


def about_page():
    """Model information page"""
    st.markdown("## 🤖 Về Mô hình")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏗️ Kiến trúc")
        st.markdown("""
        **Hybrid CNN + Vision Transformer (ViT)**
        
        Mô hình kết hợp:
        - **CNN Extractor:** 3 khối tích chập trích xuất đặc trưng cục bộ
        - **Patch Embedding:** Chuyển đổi đặc trưng CNN thành patches
        - **Vision Transformer:** Bộ mã hóa transformer 12 lớp
        - **Classifier:** Lớp tuyến tính cuối cho 9 lớp
        """)
        
        st.markdown("### 📈 Chi tiết huấn luyện")
        st.markdown("""
        - **Dataset:** ISIC 2018 Phát hiện Ung thư Da
        - **Classes:** 9 loại tổn thương da
        - **Kích thước ảnh:** 224×224 pixels
        - **Optimizer:** AdamW
        - **Hàm loss:** Focal Loss với trọng số lớp
        - **Augmentation:** Xoay, lật, thay đổi màu
        """)
    
    with col2:
        st.markdown("### 🎯 Chỉ số hiệu suất")
        st.markdown("""
        Mô hình được huấn luyện với:
        - Oversampling để cân bằng dữ liệu
        - Early stopping (patience=6)
        - Cosine annealing learning rate
        - Chọn mô hình tốt nhất dựa trên Macro F1-score
        """)
        
        st.markdown("### 📊 Thống kê mô hình")
        st.info(f"""
        - **Parameters:** ~86M (ViT-Base backbone)
        - **Thiết bị:** {DEVICE.upper()}
        - **Framework:** PyTorch + TIMM
        """)


def class_info_page():
    """Display information about all classes"""
    st.markdown("## 📚 Các loại Tổn thương Da")
    
    for i, class_name in enumerate(CLASS_NAMES):
        with st.expander(f"📌 {CLASS_NAMES_VI[i]} ({class_name})"):
            info = CLASS_INFO[class_name]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Mức độ nguy hiểm**")
                st.markdown(
                    f"<div style='background-color: {info['color']}; padding: 10px; "
                    f"border-radius: 5px; text-align: center; color: white; font-weight: bold;'>"
                    f"{info['risk']}</div>",
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown("**Mô tả**")
                st.write(info['description'])
            
            with col3:
                st.markdown("**Điều trị**")
                st.write(info['treatment'])


# ========================== RUN APP ==========================
if __name__ == "__main__":
    main()
