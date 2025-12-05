"""
HỆ THỐNG QUẢN LÝ BỆNH ÁN - CHẨN ĐOÁN UNG THƯ DA
Bệnh viện Đa khoa - Khoa Da liễu
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
from datetime import datetime
import json
from pathlib import Path

# ========================== PAGE CONFIG ==========================
st.set_page_config(
    page_title="Hệ thống Chẩn đoán Ung thư Da",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================== DATABASE ==========================
PATIENT_DB_DIR = Path("patient_database")
PATIENT_DB_DIR.mkdir(exist_ok=True)
RECORDS_FILE = PATIENT_DB_DIR / "all_records.json"

def load_all_records():
    if RECORDS_FILE.exists():
        with open(RECORDS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_record(record):
    records = load_all_records()
    records.append(record)
    with open(RECORDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def generate_patient_id():
    return f"BN{datetime.now().strftime('%Y%m%d%H%M%S')}"

def calculate_age(dob):
    today = datetime.now()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return age

# ========================== MODEL ==========================
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

# ========================== CONFIG ==========================
CHECKPOINT_PATH = "best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 9

CLASS_NAMES_EN = [
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
    'Ung thư hắc tố (Melanoma)',
    'Nốt ruồi lành tính',
    'Sừng hóa lành tính có sắc tố',
    'Sừng hóa tiết nhờn',
    'Ung thư tế bào vảy',
    'Tổn thương mạch máu da'
]

DISEASE_INFO = {
    'Actinic Keratosis': {
        'name_vi': 'Sừng hóa quang hóa',
        'icd10': 'L57.0',
        'risk': 'Trung bình',
        'description': 'Tổn thương da tiền ung thư do tiếp xúc ánh nắng mặt trời kéo dài',
        'treatment': ['Đông lạnh (Cryotherapy)', 'Thuốc bôi: 5-Fluorouracil', 'Liệu pháp quang động lực'],
        'color': '#FFA500'
    },
    'Basal Cell Carcinoma': {
        'name_vi': 'Ung thư tế bào đáy',
        'icd10': 'C44',
        'risk': 'Trung bình',
        'description': 'Loại ung thư da phổ biến nhất, phát triển chậm, hiếm khi di căn',
        'treatment': ['Phẫu thuật cắt bỏ', 'Phẫu thuật Mohs', 'Xạ trị'],
        'color': '#FF6347'
    },
    'Dermatofibroma': {
        'name_vi': 'U xơ da',
        'icd10': 'D23',
        'risk': 'Rất thấp',
        'description': 'Khối u xơ lành tính, không nguy hiểm',
        'treatment': ['Không cần điều trị', 'Phẫu thuật cắt bỏ nếu cần thiết'],
        'color': '#90EE90'
    },
    'Melanoma': {
        'name_vi': 'Ung thư hắc tố (Melanoma)',
        'icd10': 'C43',
        'risk': 'Cao',
        'description': 'Dạng ung thư da NGUY HIỂM NHẤT, có khả năng di căn cao',
        'treatment': ['⚠️ CẤP CỨU: Chuyển gấp khoa Ung thư', 'Phẫu thuật cắt rộng', 'Liệu pháp miễn dịch', 'Liệu pháp nhắm mục tiêu'],
        'color': '#DC143C'
    },
    'Nevus': {
        'name_vi': 'Nốt ruồi lành tính',
        'icd10': 'D22',
        'risk': 'Rất thấp',
        'description': 'Nốt ruồi thông thường, phần lớn lành tính',
        'treatment': ['Không cần điều trị', 'Theo dõi định kỳ', 'Cắt bỏ nếu nghi ngờ'],
        'color': '#87CEEB'
    },
    'Pigmented Benign Keratosis': {
        'name_vi': 'Sừng hóa lành tính có sắc tố',
        'icd10': 'L82',
        'risk': 'Rất thấp',
        'description': 'Tổn thương da lành tính',
        'treatment': ['Không cần điều trị y học', 'Cắt bỏ vì lý do thẩm mỹ'],
        'color': '#98FB98'
    },
    'Seborrheic Keratosis': {
        'name_vi': 'Sừng hóa tiết nhờn',
        'icd10': 'L82.1',
        'risk': 'Rất thấp',
        'description': 'Tổn thương da lành tính phổ biến ở người lớn tuổi',
        'treatment': ['Không cần điều trị', 'Đông lạnh', 'Cạo nạo điện phẫu'],
        'color': '#DDA0DD'
    },
    'Squamous Cell Carcinoma': {
        'name_vi': 'Ung thư tế bào vảy',
        'icd10': 'C44.9',
        'risk': 'Trung bình - Cao',
        'description': 'Ung thư da phổ biến thứ 2, có khả năng di căn 2-5%',
        'treatment': ['Phẫu thuật cắt bỏ', 'Phẫu thuật Mohs', 'Xạ trị', 'Liệu pháp miễn dịch'],
        'color': '#FF8C00'
    },
    'Vascular Lesion': {
        'name_vi': 'Tổn thương mạch máu da',
        'icd10': 'D18',
        'risk': 'Thấp',
        'description': 'Nhóm bệnh lý liên quan đến mạch máu da, phần lớn lành tính',
        'treatment': ['Theo dõi', 'Laser mạch máu', 'Sclerotherapy', 'Phẫu thuật'],
        'color': '#FF69B4'
    }
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    model = HybridViT(num_classes=NUM_CLASSES).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model.eval()
        return model, True
    return None, False

model, model_loaded = load_model()

def predict(image):
    if not model_loaded:
        return None, None, None, None
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    pred_idx = probabilities.argmax().item()
    pred_class_en = CLASS_NAMES_EN[pred_idx]
    pred_class_vi = CLASS_NAMES_VI[pred_idx]
    confidence = probabilities[pred_idx].item()
    
    return pred_class_en, pred_class_vi, confidence, probabilities.cpu().numpy()

def plot_probabilities_chart(probs):
    df = pd.DataFrame({
        'Bệnh': CLASS_NAMES_VI,
        'Xác suất': probs * 100
    }).sort_values('Xác suất', ascending=True)
    
    colors = ['#FF4B4B' if p < 5 else '#FFA500' if p < 15 else '#4CAF50' if p > 30 else '#2196F3' 
              for p in df['Xác suất']]
    
    fig = go.Figure(data=[
        go.Bar(
            y=df['Bệnh'],
            x=df['Xác suất'],
            orientation='h',
            text=[f'{p:.2f}%' for p in df['Xác suất']],
            textposition='outside',
            marker=dict(color=colors),
            hovertemplate='<b>%{y}</b><br>Xác suất: %{x:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Phân tích xác suất các bệnh',
        xaxis_title="Xác suất (%)",
        height=400
    )
    
    return fig

# ========================== EMERGENCY PAGE ==========================
def emergency_page():
    st.markdown("""
        <div style='background: #dc2626; padding: 1.2rem; border-radius: 8px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; font-size: 1.5rem; font-weight: 600;'>🚨 TRANG CẤP CỨU</h2>
        </div>
    """, unsafe_allow_html=True)
    
    if 'emergency_case' not in st.session_state:
        st.warning("Không có ca cấp cứu nào đang chờ xử lý.")
        return
    
    case = st.session_state['emergency_case']
    patient = case['patient_info']
    ai_diag = case['ai_diagnosis']
    
    st.markdown("""
        <div style='background: #fee2e2; border-left: 4px solid #dc2626; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;'>
            <h3 style='color: #991b1b; margin: 0 0 1rem 0;'>⚠️ CA CẤP CỨU - MỨC ĐỘ NGUY HIỂM CAO</h3>
            <p style='margin: 0.5rem 0; color: #7f1d1d; font-size: 1.1rem;'><strong>Chẩn đoán AI:</strong> {}</p>
            <p style='margin: 0.5rem 0; color: #7f1d1d;'><strong>Độ tin cậy:</strong> {}</p>
            <p style='margin: 0.5rem 0; color: #7f1d1d;'><strong>Mã ICD-10:</strong> {}</p>
        </div>
    """.format(ai_diag['disease_vi'], ai_diag['confidence'], ai_diag['icd10']), unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 👤 Thông tin bệnh nhân")
        st.markdown(f"""
            <div style='background: #f8fafc; padding: 1rem; border-radius: 8px;'>
                <p style='margin: 0.3rem 0;'><strong>🆔 Mã BN:</strong> {patient['patient_id']}</p>
                <p style='margin: 0.3rem 0;'><strong>👤 Họ tên:</strong> {patient['full_name']}</p>
                <p style='margin: 0.3rem 0;'><strong>📅 Tuổi:</strong> {patient['age']} tuổi</p>
                <p style='margin: 0.3rem 0;'><strong>⚥ Giới tính:</strong> {patient['gender']}</p>
                <p style='margin: 0.3rem 0;'><strong>📞 SĐT:</strong> {patient['phone']}</p>
                <p style='margin: 0.3rem 0;'><strong>🏠 Địa chỉ:</strong> {patient['address']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        if 'image_path' in case:
            st.markdown("### 📷 Hình ảnh tổn thương")
            img_path = PATIENT_DB_DIR / case['image_path']
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)
    
    with col2:
        st.markdown("### 🏥 Xử trí cấp cứu")
        
        st.markdown("""
            <div style='background: #fef3c7; border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
                <h4 style='color: #92400e; margin: 0 0 0.5rem 0;'>📋 Quy trình xử trí:</h4>
                <ol style='margin: 0; padding-left: 1.5rem; color: #78350f;'>
                    <li>Liên hệ ngay Khoa Ung thư</li>
                    <li>Chuẩn bị hồ sơ bệnh án đầy đủ</li>
                    <li>Chuyển gấp bệnh nhân lên khoa chuyên khoa</li>
                    <li>Thông báo gia đình về tình trạng</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
        emergency_doctor = st.text_input("👨‍⚕️ Bác sĩ cấp cứu", value="")
        oncology_contact = st.text_input("☎️ SĐT Khoa Ung thư", value="Ext: 2345")
        transfer_time = st.text_input("🕐 Thời gian chuyển khoa", value=datetime.now().strftime("%H:%M - %d/%m/%Y"))
        emergency_notes = st.text_area("📝 Ghi chú cấp cứu", height=100, 
                                       value=f"Bệnh nhân được chẩn đoán {ai_diag['disease_vi']} với độ tin cậy {ai_diag['confidence']}. Cần chuyển gấp lên Khoa Ung thư để xử trí.")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📞 Liên hệ Khoa Ung thư", type="primary", use_container_width=True):
                st.success(f"✅ Đã gọi {oncology_contact}")
        
        with col_btn2:
            if st.button("🚑 Xác nhận chuyển khoa", type="primary", use_container_width=True):
                # Cập nhật hồ sơ với thông tin cấp cứu
                case['emergency_info'] = {
                    'emergency_doctor': emergency_doctor,
                    'oncology_contact': oncology_contact,
                    'transfer_time': transfer_time,
                    'emergency_notes': emergency_notes,
                    'status': 'Đã chuyển Khoa Ung thư'
                }
                save_record(case)
                st.markdown("""
                    <div style='background: #d1fae5; border-left: 4px solid #10b981; padding: 1.2rem; border-radius: 6px; margin: 1rem 0;'>
                        <p style='margin: 0; color: #065f46; font-weight: 600; font-size: 1.1rem;'>✅ Đã xác nhận chuyển khoa thành công!</p>
                    </div>
                """, unsafe_allow_html=True)
                del st.session_state['emergency_case']
                st.rerun()

# ========================== MAIN APP ==========================
def main():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main-header {
            text-align: center;
            color: #1e40af;
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 0.3em;
            letter-spacing: -0.5px;
        }
        .sub-header {
            text-align: center;
            color: #64748b;
            font-size: 1em;
            margin-bottom: 2em;
            font-weight: 400;
        }
        
        /* Card styling */
        .stExpander {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }
        .stExpander:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        }
        
        /* Button styling */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            border: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        /* Input fields */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            border-radius: 8px;
            border: 1.5px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Metrics */
        .stMetric {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        
        /* Info/Warning boxes */
        .stAlert {
            border-radius: 10px;
            border-left: 4px solid;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Form */
        .stForm {
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        /* Section headers */
        h2, h3 {
            color: #1e293b;
            font-weight: 700;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* Image container */
        img {
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        /* Dataframe */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-header'>HỆ THỐNG CHẨN ĐOÁN UNG THƯ DA</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Bệnh viện Đa khoa - Khoa Da liễu | Hệ thống AI hỗ trợ chẩn đoán</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.5rem;'>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🏥</div>
                <h2 style='margin: 0; font-size: 1.1rem; color: #1e40af; font-weight: 600;'>MENU CHỨC NĂNG</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Hiển thị cảnh báo nếu có ca cấp cứu
        if 'emergency_case' in st.session_state:
            st.markdown("""
                <div style='background: #fee2e2; border: 2px solid #dc2626; padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem; animation: pulse 2s infinite;'>
                    <p style='margin: 0; color: #991b1b; font-weight: 600; text-align: center;'>🚨 CÓ CA CẤP CỨU!</p>
                </div>
                <style>
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.7; }
                }
                </style>
            """, unsafe_allow_html=True)
        
        page_options = ["🏠 Nhập thông tin bệnh nhân", "🔬 Chẩn đoán", "📋 Hồ sơ bệnh án", "📊 Thống kê", "🤖 Về mô hình AI"]
        if 'emergency_case' in st.session_state:
            page_options.insert(0, "🚨 TRANG CẤP CỨU")
        
        page = st.radio(
            "Chọn chức năng:",
            page_options,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("""
            <div style='background: #eff6ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 3px solid #2563eb;'>
                <h3 style='margin: 0 0 0.8rem 0; font-size: 0.95rem; color: #1e40af; font-weight: 600;'>⚙️ Thông tin hệ thống</h3>
        """, unsafe_allow_html=True)
        st.write(f"**Trạng thái:** {'🟢 Hoạt động' if model_loaded else '🔴 Lỗi'}")
        st.write(f"**Thiết bị:** {DEVICE.upper()}")
        st.write("**Mô hình:** Hybrid CNN-ViT")
        st.write("**Phiên bản:** 1.0.0")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); 
                        padding: 1rem; border-radius: 10px; border-left: 3px solid #22c55e;'>
                <h3 style='margin: 0 0 0.5rem 0; font-size: 0.95rem; color: #166534;'>💡 Mẹo sử dụng</h3>
                <p style='margin: 0; font-size: 0.85rem; color: #166534; line-height: 1.5;'>
                    • Ảnh rõ nét, đủ sáng<br>
                    • Nhập đầy đủ thông tin<br>
                    • Kiểm tra kết quả AI<br>
                    • Lưu hồ sơ định kỳ
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
            <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                        padding: 1rem; border-radius: 10px; border-left: 3px solid #f59e0b;'>
                <h3 style='margin: 0 0 0.5rem 0; font-size: 0.95rem; color: #92400e;'>📞 LIÊN HỆ</h3>
                <p style='margin: 0; font-size: 0.85rem; color: #92400e; line-height: 1.6;'>
                    <strong>Bệnh viện Đa khoa Trung ương</strong><br>
                    📍 123 Đường ABC, Quận 1, TP.HCM<br>
                    ☎️ Hotline: <strong>1900-xxxx</strong><br>
                    📧 Email: contact@hospital.vn<br>
                    🌐 Website: www.hospital.vn<br><br>
                    <strong>Khoa Da liễu</strong><br>
                    ☎️ Ext: 1234<br>
                    📧 dalieuks@hospital.vn<br>
                    ⏰ 7:30 - 17:00 (Thứ 2 - Thứ 7)
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if page == "🚨 TRANG CẤP CỨU":
        emergency_page()
    elif page == "🏠 Nhập thông tin bệnh nhân":
        patient_info_page()
    elif page == "🔬 Chẩn đoán":
        diagnosis_page()
    elif page == "📋 Hồ sơ bệnh án":
        records_page()
    elif page == "📊 Thống kê":
        statistics_page()
    elif page == "🤖 Về mô hình AI":
        model_info_page()

def patient_info_page():
    st.markdown("""
        <div style='background: #1e40af; padding: 1.2rem; border-radius: 8px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; font-size: 1.5rem; font-weight: 600;'>👤 Thông tin bệnh nhân</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("⚡ Tạo mã bệnh nhân tự động", use_container_width=True, type="secondary"):
            st.session_state['auto_patient_id'] = generate_patient_id()
            st.success(f"✨ Đã tạo mã: **{st.session_state['auto_patient_id']}**")
    
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Thông tin cơ bản**")
            patient_id = st.text_input("Mã bệnh nhân *", 
                                      value=st.session_state.get('auto_patient_id', ''),
                                      placeholder="BN202411270001")
            full_name = st.text_input("Họ và tên *", placeholder="Nguyễn Văn A")
            dob = st.date_input("Ngày sinh *")
            gender = st.selectbox("Giới tính *", ["Nam", "Nữ", "Khác"])
            phone = st.text_input("Số điện thoại *", placeholder="0912345678")
            email = st.text_input("Email", placeholder="email@example.com")
            
        with col2:
            st.markdown("**Địa chỉ liên hệ**")
            address = st.text_area("Địa chỉ", placeholder="Số nhà, đường, phường/xã")
            city = st.text_input("Tỉnh/Thành phố", placeholder="Hà Nội")
            
            st.markdown("**Thông tin BHYT**")
            insurance_id = st.text_input("Số thẻ BHYT", placeholder="GD1234567890123")
        
        st.markdown("**Tiền sử bệnh**")
        medical_history = st.multiselect("Bệnh lý mạn tính", 
            ["Không có", "Cao huyết áp", "Đái tháo đường", "Bệnh tim mạch", "Ung thư", "Dị ứng"])
        allergies = st.text_area("Dị ứng thuốc", placeholder="Liệt kê các loại dị ứng...")
        
        submitted = st.form_submit_button("💾 Lưu thông tin bệnh nhân", use_container_width=True, type="primary")
        
        if submitted:
            if patient_id and full_name and phone:
                st.session_state['patient_info'] = {
                    'patient_id': patient_id,
                    'full_name': full_name,
                    'dob': dob.strftime("%d/%m/%Y"),
                    'age': calculate_age(dob),
                    'gender': gender,
                    'phone': phone,
                    'email': email,
                    'address': address,
                    'city': city,
                    'insurance_id': insurance_id,
                    'medical_history': medical_history,
                    'allergies': allergies,
                    'created_at': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                }
                st.markdown("""
                    <div style='background: #d1fae5; border-left: 4px solid #10b981; padding: 1rem; border-radius: 6px; margin: 1rem 0;'>
                        <p style='margin: 0; color: #065f46; font-weight: 500;'>✅ Đã lưu thông tin bệnh nhân thành công!</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("⚠️ Vui lòng điền đầy đủ các trường bắt buộc (*)")

def diagnosis_page():
    st.markdown("""
        <div style='background: #059669; padding: 1.2rem; border-radius: 8px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; font-size: 1.5rem; font-weight: 600;'>🔬 Chẩn đoán tổn thương da</h2>
        </div>
    """, unsafe_allow_html=True)
    
    if 'patient_info' not in st.session_state:
        st.warning("⚠️ Chưa có thông tin bệnh nhân. Vui lòng nhập thông tin bệnh nhân trước!")
        return
    
    patient = st.session_state['patient_info']
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); 
                    padding: 1rem 1.5rem; border-radius: 10px; border-left: 4px solid #3b82f6;
                    margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            <p style='margin: 0; font-size: 1.05rem; color: #1e40af;'>
                <strong>👤 Bệnh nhân:</strong> {patient['full_name']} | 
                <strong>🆔 Mã BN:</strong> {patient['patient_id']} | 
                <strong>📅 Tuổi:</strong> {patient['age']} | 
                <strong>⚥ Giới tính:</strong> {patient['gender']}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if not model_loaded:
        st.error("❌ Lỗi: Không tải được mô hình AI. Vui lòng kiểm tra file best_model.pt")
        return
    
    uploaded_file = st.file_uploader("Tải ảnh tổn thương (JPG, PNG)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Ảnh tổn thương**")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("**Thông tin lâm sàng**")
            lesion_location = st.text_input("Vị trí tổn thương", placeholder="VD: Mặt - Trán")
            lesion_size = st.text_input("Kích thước", placeholder="VD: 5x5mm")
            clinical_notes = st.text_area("Ghi chú của bác sĩ", placeholder="Mô tả chi tiết...")
            doctor_name = st.text_input("Bác sĩ thực hiện", placeholder="BS. Nguyễn Văn A")
        
        if st.button("🔬 CHẨN ĐOÁN", use_container_width=True, type="primary"):
            with st.spinner("Đang phân tích bằng AI..."):
                pred_en, pred_vi, confidence, probs = predict(image)
            
            if pred_en:
                disease = DISEASE_INFO[pred_en]
                
                st.markdown("---")
                st.markdown("### KẾT QUẢ CHẨN ĐOÁN")
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {disease['color']}22 0%, {disease['color']}44 100%); 
                            padding: 20px; border-radius: 10px; border-left: 5px solid {disease['color']};'>
                    <h2 style='color: {disease['color']}; margin: 0;'>{pred_vi}</h2>
                    <p style='color: #666; margin: 5px 0;'><em>{pred_en}</em></p>
                    <p><strong>Mã ICD-10:</strong> {disease['icd10']} | <strong>Độ tin cậy:</strong> {confidence*100:.2f}%</p>
                    <p><strong>Mức độ nguy hiểm:</strong> {disease['risk']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Phân tích xác suất")
                fig = plot_probabilities_chart(probs)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### Thông tin bệnh")
                st.info(f"**Mô tả:** {disease['description']}")
                
                # Hiển thị phương án điều trị với cảnh báo nếu cần cấp cứu
                treatment_text = ', '.join(disease['treatment'][:2])
                if disease['risk'] == 'Cao':
                    st.markdown(f"""
                        <div style='background: #fee2e2; border-left: 4px solid #dc2626; padding: 1rem; border-radius: 6px; margin: 1rem 0;'>
                            <p style='margin: 0 0 0.5rem 0; color: #991b1b; font-weight: 600;'>⚠️ PHƯƠNG ÁN ĐIỀU TRỊ (CẤP CỨU)</p>
                            <p style='margin: 0; color: #7f1d1d;'>{treatment_text}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"**Phương án điều trị:** {treatment_text}")
                
                st.markdown("---")
                diagnosis_conclusion = st.text_area("Kết luận chẩn đoán của bác sĩ *",
                    value=f"Chẩn đoán: {pred_vi} ({pred_en}). Mã ICD-10: {disease['icd10']}. Độ tin cậy AI: {confidence*100:.1f}%.",
                    height=100)
                
                # Nút cấp cứu nếu mức độ nguy hiểm cao
                if disease['risk'] == 'Cao':
                    st.markdown("""
                        <div style='background: #fee2e2; padding: 1rem; border-radius: 8px; border: 2px solid #dc2626; margin: 1rem 0;'>
                            <p style='margin: 0 0 0.5rem 0; color: #991b1b; font-weight: 600; text-align: center;'>⚠️ TRƯỜNG HỢP CẦN XỬ TRÍ CẤP CỨU</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col_emergency, col_save = st.columns(2)
                    with col_emergency:
                        if st.button("🚨 CHUYỂN TRANG CẤP CỨU", type="primary", use_container_width=True):
                            # Lưu thông tin vào session để chuyển trang cấp cứu
                            img_filename = f"{patient['patient_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            img_path = PATIENT_DB_DIR / img_filename
                            image.save(img_path)
                            
                            st.session_state['emergency_case'] = {
                                'record_id': f"HS{datetime.now().strftime('%Y%m%d%H%M%S')}",
                                'patient_info': patient,
                                'diagnosis_date': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                'doctor': doctor_name,
                                'lesion_info': {
                                    'location': lesion_location,
                                    'size': lesion_size
                                },
                                'clinical_notes': clinical_notes,
                                'ai_diagnosis': {
                                    'disease_vi': pred_vi,
                                    'disease_en': pred_en,
                                    'confidence': f"{confidence*100:.2f}%",
                                    'icd10': disease['icd10'],
                                    'risk_level': disease['risk']
                                },
                                'doctor_conclusion': diagnosis_conclusion,
                                'treatment_plan': disease['treatment'],
                                'image_path': img_filename
                            }
                            st.rerun()
                    
                    with col_save:
                        save_button = st.button("💾 Lưu hồ sơ bệnh án", use_container_width=True)
                else:
                    save_button = st.button("💾 Lưu hồ sơ bệnh án", type="primary")
                
                if save_button:
                    img_filename = f"{patient['patient_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    img_path = PATIENT_DB_DIR / img_filename
                    image.save(img_path)
                    
                    medical_record = {
                        'record_id': f"HS{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'patient_info': patient,
                        'diagnosis_date': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        'doctor': doctor_name,
                        'lesion_info': {
                            'location': lesion_location,
                            'size': lesion_size
                        },
                        'clinical_notes': clinical_notes,
                        'ai_diagnosis': {
                            'disease_vi': pred_vi,
                            'disease_en': pred_en,
                            'confidence': f"{confidence*100:.2f}%",
                            'icd10': disease['icd10'],
                            'risk_level': disease['risk']
                        },
                        'doctor_conclusion': diagnosis_conclusion,
                        'treatment_plan': disease['treatment'],
                        'image_path': img_filename
                    }
                    
                    save_record(medical_record)
                    st.markdown(f"""
                        <div style='background: #d1fae5; border-left: 4px solid #10b981; padding: 1.2rem; border-radius: 6px; margin: 1rem 0;'>
                            <p style='margin: 0 0 0.5rem 0; color: #065f46; font-weight: 600; font-size: 1.1rem;'>✅ Đã lưu hồ sơ bệnh án thành công!</p>
                            <p style='margin: 0; color: #047857;'>📋 Mã hồ sơ: <strong>{medical_record['record_id']}</strong></p>
                        </div>
                    """, unsafe_allow_html=True)

def records_page():
    st.markdown("""
        <div style='background: #d97706; padding: 1.2rem; border-radius: 8px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; font-size: 1.5rem; font-weight: 600;'>📋 Hồ sơ bệnh án</h2>
        </div>
    """, unsafe_allow_html=True)
    
    records = load_all_records()
    
    if not records:
        st.info("📭 Chưa có hồ sơ bệnh án nào trong hệ thống.")
        return
    
    st.markdown(f"""
        <div style='background: #eff6ff; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 3px solid #2563eb;'>
            <p style='margin: 0; font-size: 1rem; color: #1e40af; font-weight: 500;'>
                📊 Tổng số hồ sơ: <span style='font-size: 1.2rem; font-weight: 600;'>{len(records)}</span>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    search_term = st.text_input("🔍 Tìm kiếm (Mã BN, Tên, Mã HS)", placeholder="Nhập từ khóa...")
    
    filtered_records = records
    if search_term:
        filtered_records = [r for r in records if 
                           search_term.lower() in r['patient_info']['patient_id'].lower() or
                           search_term.lower() in r['patient_info']['full_name'].lower() or
                           search_term.lower() in r['record_id'].lower()]
    
    if not filtered_records:
        st.warning("Không tìm thấy hồ sơ phù hợp")
        return
    
    for idx, record in enumerate(reversed(filtered_records), 1):
        patient = record['patient_info']
        ai_diag = record['ai_diagnosis']
        
        # Tạo badge màu cho mức độ nguy hiểm
        risk_colors = {
            'Cao': '#ef4444',
            'Trung bình - Cao': '#f97316',
            'Trung bình': '#f59e0b',
            'Thấp': '#84cc16',
            'Rất thấp': '#22c55e'
        }
        risk_color = risk_colors.get(ai_diag['risk_level'], '#6b7280')
        
        with st.expander(
            f"📋 Hồ sơ #{idx} | {patient['full_name']} ({patient['age']} tuổi) | {ai_diag['disease_vi']} | {record['diagnosis_date']}",
            expanded=(idx == 1)  # Mở hồ sơ mới nhất
        ):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                img_path = PATIENT_DB_DIR / record['image_path']
                if img_path.exists():
                    st.image(str(img_path), caption="Ảnh tổn thương", use_container_width=True)
                else:
                    st.warning("⚠️ Không tìm thấy ảnh")
                
                st.markdown("""
                    <div style='background: #f8fafc; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                        <h4 style='margin: 0 0 0.5rem 0; color: #1e293b;'>👤 Thông tin bệnh nhân</h4>
                    </div>
                """, unsafe_allow_html=True)
                st.write(f"**🆔 Mã BN:** {patient['patient_id']}")
                st.write(f"**👤 Họ tên:** {patient['full_name']}")
                st.write(f"**📅 Tuổi:** {patient['age']} tuổi")
                st.write(f"**⚥ Giới tính:** {patient['gender']}")
                st.write(f"**📞 SĐT:** {patient.get('phone', 'N/A')}")
                st.write(f"**📧 Email:** {patient.get('email', 'N/A')}")
                st.write(f"**🏠 Địa chỉ:** {patient.get('address', 'N/A')}")
                st.write(f"**🏙️ Thành phố:** {patient.get('city', 'N/A')}")
                
                if patient.get('insurance_id'):
                    st.write(f"**🆔 BHYT:** {patient['insurance_id']}")
                
                if patient.get('medical_history'):
                    st.write(f"**🏥 Tiền sử:** {', '.join(patient['medical_history'])}")
            
            with col2:
                disease_info = DISEASE_INFO.get(ai_diag['disease_en'], {})
                
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {disease_info.get('color', '#3b82f6')}22 0%, {disease_info.get('color', '#3b82f6')}44 100%); 
                                padding: 1.5rem; border-radius: 12px; border-left: 5px solid {disease_info.get('color', '#3b82f6')};'>
                        <h3 style='color: {disease_info.get('color', '#3b82f6')}; margin: 0 0 0.5rem 0;'>{ai_diag['disease_vi']}</h3>
                        <p style='color: #64748b; margin: 0 0 1rem 0; font-style: italic;'>{ai_diag['disease_en']}</p>
                        <p style='margin: 0.3rem 0;'><strong>🏷️ Mã ICD-10:</strong> <code>{ai_diag['icd10']}</code></p>
                        <p style='margin: 0.3rem 0;'><strong>🎯 Độ tin cậy:</strong> <span style='font-size: 1.2rem; font-weight: bold; color: {disease_info.get('color', '#3b82f6')};'>{ai_diag['confidence']}</span></p>
                        <p style='margin: 0.3rem 0;'><strong>⚠️ Mức độ:</strong> 
                            <span style='background: {risk_color}; color: white; padding: 0.2rem 0.8rem; border-radius: 20px; font-weight: 600;'>
                                {ai_diag['risk_level']}
                            </span>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("")
                st.markdown("**📝 Kết luận của bác sĩ:**")
                st.info(record['doctor_conclusion'])
                
                st.markdown("**👨‍⚕️ Thông tin khám:**")
                st.write(f"• Bác sĩ: {record.get('doctor', 'N/A')}")
                st.write(f"• Ngày khám: {record['diagnosis_date']}")
                st.write(f"• Mã hồ sơ: {record['record_id']}")
                
                lesion = record.get('lesion_info', {})
                if lesion:
                    st.markdown("**📍 Thông tin tổn thương:**")
                    st.write(f"• Vị trí: {lesion.get('location', 'N/A')}")
                    st.write(f"• Kích thước: {lesion.get('size', 'N/A')}")
                
                if record.get('clinical_notes'):
                    st.markdown("**📋 Ghi chú lâm sàng:**")
                    st.write(record['clinical_notes'])

def model_info_page():
    st.markdown("""
        <div style='background: #0891b2; padding: 1.2rem; border-radius: 8px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; font-size: 1.5rem; font-weight: 600;'>🤖 Giới thiệu mô hình AI</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Tổng quan
    st.markdown("""
        <div style='background: #f0f9ff; border-left: 4px solid #0891b2; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;'>
            <h3 style='color: #0c4a6e; margin: 0 0 1rem 0;'>📋 Tổng quan</h3>
            <p style='margin: 0.5rem 0; color: #0c4a6e; line-height: 1.8;'>
                Hệ thống sử dụng mô hình <strong>Hybrid CNN-ViT (Convolutional Neural Network + Vision Transformer)</strong> 
                để phân loại 9 loại tổn thương da với độ chính xác cao. Mô hình kết hợp ưu điểm của CNN trong trích xuất 
                đặc trưng cục bộ và ViT trong học các mối quan hệ toàn cục của hình ảnh.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Kiến trúc mô hình
    st.markdown("### 🏗️ Kiến trúc mô hình")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
            <div style='background: white; border: 2px solid #e0f2fe; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem;'>
                <h4 style='color: #0891b2; margin: 0 0 0.8rem 0;'>🔷 CNN Extractor</h4>
                <p style='margin: 0.3rem 0; color: #334155; font-size: 0.9rem;'>
                    <strong>• Conv Block 1:</strong> 3 → 32 channels<br>
                    <strong>• Conv Block 2:</strong> 32 → 64 channels<br>
                    <strong>• Conv Block 3:</strong> 64 → 128 channels<br>
                    <strong>• Kích hoạt:</strong> ReLU + BatchNorm<br>
                    <strong>• Pooling:</strong> MaxPool2d (2x2)
                </p>
                <p style='margin: 0.8rem 0 0 0; color: #64748b; font-size: 0.85rem; font-style: italic;'>
                    👉 Trích xuất đặc trưng cục bộ từ ảnh đầu vào
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: white; border: 2px solid #ddd6fe; padding: 1.2rem; border-radius: 8px;'>
                <h4 style='color: #7c3aed; margin: 0 0 0.8rem 0;'>🔷 Vision Transformer</h4>
                <p style='margin: 0.3rem 0; color: #334155; font-size: 0.9rem;'>
                    <strong>• Architecture:</strong> ViT-Base<br>
                    <strong>• Transformer Layers:</strong> 12 layers<br>
                    <strong>• Embedding Dim:</strong> 768<br>
                    <strong>• Attention Heads:</strong> 12 heads<br>
                    <strong>• MLP Ratio:</strong> 4x
                </p>
                <p style='margin: 0.8rem 0 0 0; color: #64748b; font-size: 0.85rem; font-style: italic;'>
                    👉 Học mối quan hệ toàn cục giữa các vùng ảnh
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: white; border: 2px solid #fed7aa; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem;'>
                <h4 style='color: #ea580c; margin: 0 0 0.8rem 0;'>🔷 Patch Embedding</h4>
                <p style='margin: 0.3rem 0; color: #334155; font-size: 0.9rem;'>
                    <strong>• Input:</strong> 128 channels từ CNN<br>
                    <strong>• Output:</strong> 768-dim embeddings<br>
                    <strong>• Patch Size:</strong> 2x2<br>
                    <strong>• Method:</strong> Conv2d projection
                </p>
                <p style='margin: 0.8rem 0 0 0; color: #64748b; font-size: 0.85rem; font-style: italic;'>
                    👉 Chuyển đổi feature maps thành sequence
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: white; border: 2px solid #bbf7d0; padding: 1.2rem; border-radius: 8px;'>
                <h4 style='color: #15803d; margin: 0 0 0.8rem 0;'>🔷 Classifier</h4>
                <p style='margin: 0.3rem 0; color: #334155; font-size: 0.9rem;'>
                    <strong>• Input:</strong> 768-dim CLS token<br>
                    <strong>• Output:</strong> 9 classes<br>
                    <strong>• Type:</strong> Linear layer<br>
                    <strong>• Activation:</strong> Softmax
                </p>
                <p style='margin: 0.8rem 0 0 0; color: #64748b; font-size: 0.85rem; font-style: italic;'>
                    👉 Dự đoán xác suất cho từng loại bệnh
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Phân bổ parameters
    st.markdown("### 🔢 Phân bổ Parameters")
    
    st.markdown("""
        <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 2px solid #0891b2; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h4 style='color: #0c4a6e; margin: 0 0 1rem 0; text-align: center;'>📊 Tổng số Parameters trong Mô hình</h4>
            <div style='background: white; padding: 1.2rem; border-radius: 8px;'>
                <table style='width: 100%; border-collapse: collapse;'>
                    <thead>
                        <tr style='background: #f1f5f9; border-bottom: 2px solid #cbd5e1;'>
                            <th style='padding: 0.8rem; text-align: left; color: #334155; font-weight: 600;'>Module</th>
                            <th style='padding: 0.8rem; text-align: right; color: #334155; font-weight: 600;'>Parameters</th>
                            <th style='padding: 0.8rem; text-align: right; color: #334155; font-weight: 600;'>Tỷ lệ</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style='border-bottom: 1px solid #e2e8f0;'>
                            <td style='padding: 0.8rem; color: #475569;'>🔷 CNN Extractor</td>
                            <td style='padding: 0.8rem; text-align: right; color: #0891b2; font-weight: 600;'>~57,000</td>
                            <td style='padding: 0.8rem; text-align: right; color: #64748b;'>0.07%</td>
                        </tr>
                        <tr style='border-bottom: 1px solid #e2e8f0;'>
                            <td style='padding: 0.8rem; color: #475569;'>🔷 Patch Embedding</td>
                            <td style='padding: 0.8rem; text-align: right; color: #ea580c; font-weight: 600;'>~196,000</td>
                            <td style='padding: 0.8rem; text-align: right; color: #64748b;'>0.23%</td>
                        </tr>
                        <tr style='border-bottom: 1px solid #e2e8f0; background: #fef3c7;'>
                            <td style='padding: 0.8rem; color: #78350f; font-weight: 600;'>🔷 ViT Transformer</td>
                            <td style='padding: 0.8rem; text-align: right; color: #7c3aed; font-weight: 700;'>~85,800,000</td>
                            <td style='padding: 0.8rem; text-align: right; color: #92400e; font-weight: 600;'>99.7%</td>
                        </tr>
                        <tr style='border-bottom: 1px solid #e2e8f0;'>
                            <td style='padding: 0.8rem; color: #475569;'>🔷 Classifier Head</td>
                            <td style='padding: 0.8rem; text-align: right; color: #15803d; font-weight: 600;'>~6,921</td>
                            <td style='padding: 0.8rem; text-align: right; color: #64748b;'>0.008%</td>
                        </tr>
                        <tr style='background: #dbeafe; border-top: 3px solid #0891b2;'>
                            <td style='padding: 1rem; color: #0c4a6e; font-weight: 700; font-size: 1.05rem;'>📌 TỔNG CỘNG</td>
                            <td style='padding: 1rem; text-align: right; color: #0891b2; font-weight: 700; font-size: 1.2rem;'>~86,054,000</td>
                            <td style='padding: 1rem; text-align: right; color: #0c4a6e; font-weight: 700;'>100%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div style='background: #fef3c7; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #f59e0b;'>
                <p style='margin: 0; color: #78350f; font-size: 0.95rem;'>
                    <strong>💡 Lưu ý:</strong> ViT Transformer chiếm gần như toàn bộ parameters (99.7%), 
                    cho thấy khả năng học representation mạnh mẽ từ pretrained ImageNet. 
                    CNN Extractor và Patch Embedding chỉ chiếm 0.3% nhưng đóng vai trò quan trọng 
                    trong việc tiền xử lý và chuẩn bị dữ liệu cho Transformer.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Thông số mô hình
    st.markdown("### 📊 Thông số mô hình")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); padding: 1.5rem; border-radius: 8px; text-align: center;'>
                <p style='margin: 0; color: white; font-size: 2rem; font-weight: 700;'>86M</p>
                <p style='margin: 0.3rem 0 0 0; color: #dbeafe; font-size: 0.9rem;'>Parameters</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 1.5rem; border-radius: 8px; text-align: center;'>
                <p style='margin: 0; color: white; font-size: 2rem; font-weight: 700;'>224×224</p>
                <p style='margin: 0.3rem 0 0 0; color: #d1fae5; font-size: 0.9rem;'>Input Size</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); padding: 1.5rem; border-radius: 8px; text-align: center;'>
                <p style='margin: 0; color: white; font-size: 2rem; font-weight: 700;'>9</p>
                <p style='margin: 0.3rem 0 0 0; color: #fef3c7; font-size: 0.9rem;'>Classes</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); padding: 1.5rem; border-radius: 8px; text-align: center;'>
                <p style='margin: 0; color: white; font-size: 2rem; font-weight: 700;'>ISIC</p>
                <p style='margin: 0.3rem 0 0 0; color: #ede9fe; font-size: 0.9rem;'>Dataset</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quá trình huấn luyện
    st.markdown("### 🎯 Quá trình huấn luyện")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
            <div style='background: #fef3c7; border-left: 4px solid #f59e0b; padding: 1.2rem; border-radius: 8px;'>
                <h4 style='color: #92400e; margin: 0 0 0.8rem 0;'>⚙️ Cấu hình huấn luyện</h4>
                <p style='margin: 0.3rem 0; color: #78350f; font-size: 0.9rem;'>
                    <strong>• Optimizer:</strong> AdamW (lr=3e-4)<br>
                    <strong>• Loss Function:</strong> Focal Loss (γ=2.0)<br>
                    <strong>• Scheduler:</strong> CosineAnnealingLR<br>
                    <strong>• Batch Size:</strong> 32<br>
                    <strong>• Early Stopping:</strong> Patience = 6<br>
                    <strong>• Data Augmentation:</strong> Flip, Rotate, ColorJitter<br>
                    <strong>• Oversampling:</strong> 5x cho class thiểu số
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: #dcfce7; border-left: 4px solid #22c55e; padding: 1.2rem; border-radius: 8px;'>
                <h4 style='color: #166534; margin: 0 0 0.8rem 0;'>📈 Kết quả</h4>
                <p style='margin: 0.3rem 0; color: #14532d; font-size: 0.9rem;'>
                    <strong>• Training Accuracy:</strong> > 95%<br>
                    <strong>• Validation Accuracy:</strong> > 90%<br>
                    <strong>• Macro F1-Score:</strong> > 0.88<br>
                    <strong>• Inference Time:</strong> < 0.5s/image<br>
                    <strong>• Model Selection:</strong> Best macro F1<br>
                    <strong>• Dataset:</strong> ISIC 2018 Skin Lesions
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # 9 loại bệnh
    st.markdown("### 🏥 Các loại bệnh được phát hiện")
    
    diseases_data = [
        ["Actinic Keratosis (Sừng hóa quang hóa)", "L57.0", "Trung bình", "Tiền ung thư"],
        ["Basal Cell Carcinoma (Ung thư tế bào đáy)", "C44", "Trung bình", "Ung thư da phổ biến nhất"],
        ["Dermatofibroma (U xơ da)", "D23", "Rất thấp", "Lành tính"],
        ["Melanoma (Ung thư hắc tố)", "C43", "Cao", "Nguy hiểm nhất"],
        ["Nevus (Nốt ruồi lành tính)", "D22", "Rất thấp", "Lành tính"],
        ["Pigmented Benign Keratosis (Sừng hóa có sắc tố)", "L82", "Rất thấp", "Lành tính"],
        ["Seborrheic Keratosis (Sừng hóa tiết nhờn)", "L82.1", "Rất thấp", "Lành tính"],
        ["Squamous Cell Carcinoma (Ung thư tế bào vảy)", "C44.9", "Trung bình - Cao", "Ung thư da thứ 2"],
        ["Vascular Lesion (Tổn thương mạch máu)", "D18", "Thấp", "Liên quan mạch máu"]
    ]
    
    df = pd.DataFrame(diseases_data, columns=["Tên bệnh", "ICD-10", "Mức độ nguy hiểm", "Đặc điểm"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Lưu ý sử dụng
    st.markdown("### ⚠️ Lưu ý quan trọng")
    st.markdown("""
        <div style='background: #fee2e2; border-left: 4px solid #dc2626; padding: 1.2rem; border-radius: 8px;'>
            <p style='margin: 0.5rem 0; color: #7f1d1d; line-height: 1.8;'>
                <strong>• Công cụ hỗ trợ:</strong> Kết quả AI chỉ mang tính chất tham khảo, KHÔNG thay thế chẩn đoán của bác sĩ chuyên khoa.<br>
                <strong>• Xác nhận bởi bác sĩ:</strong> Mọi quyết định điều trị phải được bác sĩ da liễu xác nhận.<br>
                <strong>• Chất lượng ảnh:</strong> Kết quả phụ thuộc vào độ rõ nét, ánh sáng của ảnh đầu vào.<br>
                <strong>• Cập nhật liên tục:</strong> Mô hình được cải tiến và cập nhật định kỳ để nâng cao độ chính xác.
            </p>
        </div>
    """, unsafe_allow_html=True)

def statistics_page():
    st.markdown("""
        <div style='background: #7c3aed; padding: 1.2rem; border-radius: 8px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; font-size: 1.5rem; font-weight: 600;'>📊 Thống kê báo cáo</h2>
        </div>
    """, unsafe_allow_html=True)
    
    records = load_all_records()
    
    if not records:
        st.info("Chưa có dữ liệu để thống kê")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        unique_patients = len(set(r['patient_info']['patient_id'] for r in records))
        st.metric("Tổng bệnh nhân", unique_patients)
    
    with col2:
        st.metric("Tổng hồ sơ", len(records))
    
    with col3:
        diseases = [r['ai_diagnosis']['disease_vi'] for r in records]
        most_common = max(set(diseases), key=diseases.count) if diseases else "N/A"
        st.metric("Bệnh phổ biến", most_common)
    
    with col4:
        high_risk = sum(1 for r in records if r['ai_diagnosis']['risk_level'] in ['Cao', 'Trung bình - Cao'])
        st.metric("Ca nguy hiểm", high_risk)
    
    st.markdown("---")
    st.markdown("### Phân bố các loại bệnh")
    
    disease_counts = {}
    for record in records:
        disease = record['ai_diagnosis']['disease_vi']
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    df_diseases = pd.DataFrame({
        'Bệnh': list(disease_counts.keys()),
        'Số ca': list(disease_counts.values())
    }).sort_values('Số ca', ascending=False)
    
    fig = go.Figure(data=[
        go.Bar(x=df_diseases['Số ca'], y=df_diseases['Bệnh'], orientation='h', marker=dict(color='#3b82f6'))
    ])
    fig.update_layout(title="Số lượng ca theo từng loại bệnh", xaxis_title="Số ca", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(df_diseases, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
