"""
Professional Streamlit Web Interface for Skin Cancer Classification
Model: HybridViT (CNN + Vision Transformer)
Version: 3.0 - Professional UI with Blue Theme
"""

import os
import time
import torch
import torch.nn as nn
import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import pandas as pd
import timm
from torchvision import transforms
import plotly.graph_objects as go

# ========================== PAGE CONFIG ==========================
st.set_page_config(
    page_title="Skin Cancer AI Detection",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================== LANGUAGE SETTINGS ==========================
# Initialize language in session state
if 'language' not in st.session_state:
    st.session_state.language = 'vi'  # Default to Vietnamese

# Language dictionary
TRANSLATIONS = {
    'vi': {
        'title': 'HỆ THỐNG PHÁT HIỆN UNG THƯ DA BẰNG AI',
        'subtitle': 'Phân loại tổn thương da với HybridViT (CNN + Vision Transformer)',
        'upload_title': 'TẢI ẢNH LÊN',
        'upload_help': 'Tải ảnh da cần phân tích (JPG, PNG)',
        'analyzing': 'Đang phân tích ảnh...',
        'prediction_result': 'KẾT QUẢ DỰ ĐOÁN',
        'confidence': 'Độ tin cậy',
        'top5_predictions': 'TOP 5 DỰ ĐOÁN',
        'disease_info': 'THÔNG TIN VỀ',
        'consult_doctor': 'QUAN TRỌNG: Kết quả chỉ mang tính tham khảo. Luôn tham khảo bác sĩ da liễu!',
        'system_info': 'THÔNG TIN HỆ THỐNG',
        'model_version': 'Phiên bản',
        'architecture': 'Kiến trúc',
        'dataset': 'Dataset',
        'accuracy': 'Độ chính xác',
        'classes': 'Số lớp',
        'how_to_use': 'HƯỚNG DẪN SỬ DỤNG',
        'step1': 'Tải ảnh da lên hệ thống',
        'step2': 'AI tự động phân tích và nhận diện',
        'step3': 'Xem kết quả, biểu đồ và thông tin chi tiết',
        'step4': 'Tham khảo bác sĩ để chẩn đoán chuyên sâu',
        'model_info': 'THÔNG TIN MODEL',
        'warning': 'LƯU Ý Y TẾ',
        'warning_text': 'Ứng dụng này CHỈ hỗ trợ tham khảo, KHÔNG thay thế chẩn đoán y khoa chuyên nghiệp. Luôn tham khảo bác sĩ da liễu có chứng chỉ!',
        # Flowchart
        'workflow_title': 'Quy trình phân tích AI',
        'flow_step1': 'Chuẩn bị ảnh',
        'flow_step2': 'Tải ảnh lên',
        'flow_step3': 'AI phân tích',
        'flow_step4': 'Nhận kết quả',
        'flow_step5': 'Tham khảo bác sĩ',
        # Disease types
        'diseases_title': '⚕ Hệ thống có thể phát hiện 9 loại tổn thương da',
        # Sidebar
        'system_ai': 'HỆ THỐNG AI',
        'device': 'Thiết bị:',
        'status': 'Trạng thái:',
        'ready': 'Sẵn sàng',
        'guide_title': 'HƯỚNG DẪN SỬ DỤNG',
        'guide_step1': 'Tải ảnh tổn thương da lên hệ thống',
        'guide_step2': 'AI tự động phân tích và nhận diện',
        'guide_step3': 'Xem kết quả, biểu đồ và thông tin chi tiết',
        'guide_step4': 'Tham khảo bác sĩ để chẩn đoán chuyên sâu',
        'important_note': 'LƯU Ý QUAN TRỌNG',
        'note_text': 'Kết quả AI chỉ mang tính chất tham khảo. Luôn tham khảo ý kiến bác sĩ chuyên khoa để có chẩn đoán chính xác.',
        'cannot_load_model': 'Không thể tải model từ:',
        'ensure_model': 'Vui lòng đảm bảo file \'best_model.pt\' có trong thư mục gốc.',
        # Gemini advice
        'advice_title': 'Tư vấn chăm sóc (AI)',
        'advice_desc': 'Gợi ý hành động an toàn dựa trên dự đoán AI. KHÔNG thay thế tư vấn y khoa và KHÔNG kê thuốc.',
        'advice_textarea': 'Mô tả thêm triệu chứng / thời gian xuất hiện (tùy chọn)',
        'advice_btn': 'Lấy gợi ý chăm sóc',
        'advice_loading': 'Đang lấy gợi ý từ Gemini...',
        'advice_result_title': 'Gợi ý hành động',
        'advice_missing_key': 'Chưa thiết lập GEMINI_API_KEY. Vui lòng thêm vào biến môi trường hoặc st.secrets.',
        'advice_error': 'Không thể lấy gợi ý từ Gemini. Vui lòng thử lại.',
        'advice_disclaimer': 'Gợi ý chỉ mang tính tham khảo, không chẩn đoán hay kê đơn. Luôn gặp bác sĩ chuyên khoa.'
    },
    'en': {
        'title': 'AI-POWERED SKIN CANCER DETECTION SYSTEM',
        'subtitle': 'Skin Lesion Classification with HybridViT (CNN + Vision Transformer)',
        'upload_title': 'UPLOAD IMAGE',
        'upload_help': 'Upload skin image for analysis (JPG, PNG)',
        'analyzing': 'Analyzing image...',
        'prediction_result': 'PREDICTION RESULT',
        'confidence': 'Confidence',
        'top5_predictions': 'TOP 5 PREDICTIONS',
        'disease_info': 'INFORMATION ABOUT',
        'consult_doctor': 'IMPORTANT: Results are for reference only. Always consult a dermatologist!',
        'system_info': 'SYSTEM INFORMATION',
        'model_version': 'Version',
        'architecture': 'Architecture',
        'dataset': 'Dataset',
        'accuracy': 'Accuracy',
        'classes': 'Classes',
        'how_to_use': 'HOW TO USE',
        'step1': 'Upload skin image to system',
        'step2': 'AI automatically analyzes and identifies',
        'step3': 'View results, charts and detailed information',
        'step4': 'Consult doctor for professional diagnosis',
        'model_info': 'MODEL INFORMATION',
        'warning': 'MEDICAL DISCLAIMER',
        'warning_text': 'This application is for REFERENCE ONLY and does NOT replace professional medical diagnosis. Always consult a certified dermatologist!',
        # Flowchart
        'workflow_title': 'AI Analysis Workflow',
        'flow_step1': 'Prepare Image',
        'flow_step2': 'Upload Image',
        'flow_step3': 'AI Analysis',
        'flow_step4': 'Get Results',
        'flow_step5': 'Consult Doctor',
        # Disease types
        'diseases_title': '⚕ System can detect 9 types of skin lesions',
        # Sidebar
        'system_ai': 'AI SYSTEM',
        'device': 'Device:',
        'status': 'Status:',
        'ready': 'Ready',
        'guide_title': 'USER GUIDE',
        'guide_step1': 'Upload skin lesion image to system',
        'guide_step2': 'AI automatically analyzes and identifies',
        'guide_step3': 'View results, charts and detailed information',
        'guide_step4': 'Consult doctor for professional diagnosis',
        'important_note': 'IMPORTANT NOTE',
        'note_text': 'AI results are for reference only. Always consult a specialist for accurate diagnosis.',
        'cannot_load_model': 'Cannot load model from:',
        'ensure_model': 'Please ensure \'best_model.pt\' file exists in the root directory.',
        # Gemini advice
        'advice_title': 'Care advice (AI)',
        'advice_desc': 'Safety-first guidance based on the AI prediction. Not medical advice and never prescriptive.',
        'advice_textarea': 'Describe additional symptoms / duration (optional)',
        'advice_btn': 'Get care suggestions',
        'advice_loading': 'Fetching suggestions from Gemini...',
        'advice_result_title': 'Suggested actions',
        'advice_missing_key': 'GEMINI_API_KEY is not set. Please provide via environment variable or st.secrets.',
        'advice_error': 'Unable to fetch advice from Gemini. Please try again.',
        'advice_disclaimer': 'This is reference-only guidance, not a diagnosis or prescription. Always see a specialist.'
    }
}

def t(key: str) -> str:
    """Get translated text based on current language"""
    lang = st.session_state.get('language', 'vi')
    return TRANSLATIONS.get(lang, TRANSLATIONS['vi']).get(key, key)

# ========================== MODEL ARCHITECTURE ==========================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return x * out


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class CNNExtractorCBAM(nn.Module):
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
        self.cbam = CBAM(128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.cbam(x)
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
        self.cnn = CNNExtractorCBAM()
        self.patch_embed = PatchEmbed()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
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
# Get the absolute path to ensure model is found
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "best_model_CNN_CBAM_ViT.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 9
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

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


# ========================== DOWNLOAD MODEL ==========================
def download_model_from_drive():
    """Download model from Google Drive with progress bar"""
    import gdown
    import requests
    from tqdm import tqdm
    
    # Google Drive file ID từ link
    file_id = "1QGJOCE4DIaqbj5DfMmfXoYL8D20xJ8XI"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Thêm spacing để tránh bị header che
    st.write("")
    st.write("")
    
    st.toast("⬇ Đang tải model từ Google Drive... (330MB)")
    progress_bar = st.progress(0)
    
    try:
        # Download với gdown
        output = gdown.download(id=file_id, output=CHECKPOINT_PATH, quiet=False)
        
        if output and os.path.exists(CHECKPOINT_PATH):
            progress_bar.progress(100)
            file_size = os.path.getsize(CHECKPOINT_PATH) / (1024**2)
            st.toast(f"✓ Tải model thành công! ({file_size:.1f}MB)")
            time.sleep(10)  # Hiển thị 10s
            progress_bar.empty()
            return True
        else:
            st.toast("Không thể tải model. Vui lòng thử lại.")
            return False
        
    except Exception as e:
        st.toast(f"Lỗi: {str(e)}")
        return False


# ========================== LOAD MODEL ==========================
@st.cache_resource(show_spinner=False)
def _load_model_from_checkpoint(checkpoint_path, device, num_classes):
    """Internal function to load model from checkpoint file - cached"""
    model = HybridViT(num_classes=num_classes).to(device)
    
    # Try loading without weights_only first (for older PyTorch versions)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Checkpoint có thể là state_dict trực tiếp hoặc dictionary có key 'model_state_dict'
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Checkpoint là state_dict trực tiếp
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def load_model(checkpoint_path, device, num_classes):
    """Load model with auto-download if needed"""
    
    # Kiểm tra file có tồn tại không
    if not os.path.exists(checkpoint_path):
        # Tải từ Google Drive nếu chưa có
        download_success = download_model_from_drive()
        if not download_success:
            return None, False, "download_failed"
    
    # Load model từ checkpoint (cached)
    try:
        model = _load_model_from_checkpoint(checkpoint_path, device, num_classes)
        return model, True, "loaded"
    except Exception as e:
        return None, False, f"error: {str(e)}"


# Load model và hiển thị thông báo
st.write("")
st.write("")

with st.spinner("Đang load model..."):
    model, model_loaded, load_status = load_model(CHECKPOINT_PATH, DEVICE, NUM_CLASSES)

# Hiển thị toast dựa trên kết quả (auto-hide sau 10s)
if model_loaded:
    st.toast("✓ Model đã sẵn sàng!")
elif "error" in load_status:
    st.toast(f"❌ {load_status.replace('error: ', '')}")
elif load_status == "download_failed":
    st.toast("❌ Không thể tải model từ Google Drive")


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


def generate_gemini_advice(pred_class, pred_class_vi, risk, confidence, user_notes, lang_code):
    """Call Gemini to produce safety-first care advice (no prescriptions)."""
    if not GEMINI_API_KEY:
        return None, 'missing_key'
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Try different model names that may be available (newest to oldest)
        model_names_to_try = [
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro",
            "gemini-3.0",
        ]
        
        model = None
        last_error = None
        for model_name in model_names_to_try:
            try:
                test_model = genai.GenerativeModel(model_name)
                # Try a simple test to verify the model works
                test_response = test_model.generate_content("test")
                model = test_model
                break
            except Exception as e:
                last_error = str(e)
                continue
        
        if model is None:
            # Fallback: try to get first available model
            try:
                available_models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                if available_models:
                    model_name = available_models[0].name.replace('models/', '')
                    model = genai.GenerativeModel(model_name)
            except Exception as e:
                last_error = str(e)
        
        if model is None:
            error_msg = f"Không thể kết nối Gemini API. Lỗi: {last_error}"
            return None, error_msg
        
        lang_label = 'Vietnamese' if lang_code == 'vi' else 'English'
        prompt = f"""
You are a dermatology decision-support assistant. Provide concise, safety-first guidance.
Inputs:
- Predicted condition: {pred_class} ({pred_class_vi})
- Risk tier: {risk}
- Model confidence: {confidence*100:.1f}%
- User-described symptoms: {user_notes or 'N/A'}
Output language: {lang_label}
Rules:
- Do NOT prescribe or name specific drugs or treatments.
- Do NOT give detailed self-treatment steps; emphasize seeing a specialist for procedures/meds.
- Structure the reply in 4 short bullet points: (1) severity & urgency, (2) immediate self-care basics (non-pharmacologic), (3) when to see a dermatologist, (4) red-flag signs that require urgent care.
- Keep total under 120 words.
"""
        response = model.generate_content(prompt)
        return response.text, None
    except Exception as e:
        return None, str(e)


# ========================== VISUALIZATION FUNCTIONS ==========================
def plot_probabilities_chart(probs, class_names_vi):
    """Create horizontal bar chart"""
    df = pd.DataFrame({
        'Class': class_names_vi,
        'Probability': probs * 100
    }).sort_values('Probability', ascending=True)
    
    colors = ['#E3F2FD' if p < 10 else '#90CAF9' if p < 30 else '#42A5F5' if p < 60 else '#1976D2' 
              for p in df['Probability']]
    
    fig = go.Figure(data=[
        go.Bar(
            y=df['Class'],
            x=df['Probability'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='#1565C0', width=1.5),
                cornerradius=8
            ),
            text=[f'{p:.1f}%' for p in df['Probability']],
            textposition='outside',
            textfont=dict(size=13, family='Inter', color='#0D47A1'),
            hovertemplate='<b>%{y}</b><br>Xác suất: %{x:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        xaxis_title="Xác suất (%)",
        yaxis_title="",
        height=380,
        font=dict(size=12, family='Inter', color='#0D47A1'),
        plot_bgcolor='rgba(227,242,253,0.2)',
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
        margin=dict(l=20, r=120, t=20, b=50)
    )
    
    return fig


def plot_confidence_gauge(confidence):
    """Create confidence gauge"""
    color = '#E53935' if confidence < 0.5 else '#FB8C00' if confidence < 0.7 else '#1976D2'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 56, 'color': '#0D47A1'}},
        gauge={
            'axis': {
                'range': [None, 100], 
                'tickwidth': 2,
                'tickcolor': '#1976D2',
                'tickfont': {'size': 14, 'color': '#1565C0'}
            },
            'bar': {'color': color, 'thickness': 0.85},
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
        height=260,
        margin=dict(l=30, r=30, t=10, b=10),
        paper_bgcolor='rgba(227,242,253,0.2)',
        font={'family': 'Inter'}
    )
    
    return fig


def plot_top_predictions(probs, class_names_vi, top_n=5):
    """Create donut chart for top predictions"""
    top_idx = np.argsort(probs)[::-1][:top_n]
    top_probs = probs[top_idx]
    top_names = [class_names_vi[i] for i in top_idx]
    
    colors = ['#0D47A1', '#1565C0', '#1976D2', '#1E88E5', '#42A5F5']
    
    fig = go.Figure(data=[
        go.Pie(
            labels=top_names,
            values=top_probs * 100,
            hole=0.65,
            marker=dict(
                colors=colors, 
                line=dict(color='white', width=3)
            ),
            textfont=dict(size=14, family='Inter', color='white'),
            textposition='inside',
            hovertemplate='<b>%{label}</b><br>%{value:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        height=320,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=13, family='Inter', color='#1565C0'),
            bgcolor='rgba(227,242,253,0.3)',
            bordercolor='#1976D2',
            borderwidth=1
        ),
        margin=dict(l=30, r=30, t=20, b=20),
        paper_bgcolor='rgba(227,242,253,0.2)'
    )
    
    return fig


# ========================== CUSTOM CSS ==========================
def load_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        .main {
            background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
            font-family: 'Inter', sans-serif;
        }
        
        .block-container {
            padding: 1.5rem 1.5rem;
            background: linear-gradient(to bottom, #ffffff 0%, #f8fbff 100%);
            border-radius: 20px;
            margin: 15px;
            box-shadow: 0 15px 40px rgba(13,71,161,0.12);
            max-width: 1400px;
        }
        
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1976D2 0%, #0D47A1 100%);
            width: 25rem !important;
            min-width: 25rem !important;
            max-width: 25rem !important;
        }
        
        section[data-testid="stSidebar"] * {
            color: white !important;
        }
        
        section[data-testid="stSidebar"] > div:first-child {
            width: 25rem !important;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 16px 40px;
            font-weight: 700;
            font-size: 16px;
            transition: all 0.3s;
            box-shadow: 0 8px 24px rgba(25,118,210,0.35);
            letter-spacing: 0.8px;
            text-transform: uppercase;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 32px rgba(25,118,210,0.45);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(227,242,253,0.3);
            padding: 10px;
            border-radius: 14px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 14px 28px;
            font-weight: 600;
            color: #1565C0;
            font-size: 15px;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
            color: white !important;
        }
        
        img {
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        }
        
        hr {
            margin: 1rem 0;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #1976D2, transparent);
        }
        
        .stAlert {
            border-radius: 12px;
            border-left: 4px solid;
            padding: 12px 16px;
        }
        
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
        
        code {
            background: rgba(227,242,253,0.5);
            color: #0D47A1;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
        }
        
        .metric-card {
            background: white;
            padding: 24px;
            border-radius: 16px;
            border-left: 5px solid #1976D2;
            box-shadow: 0 4px 12px rgba(25,118,210,0.1);
            margin: 12px 0;
        }
        
        .section-header {
            text-align: center;
            padding: 14px;
            background: linear-gradient(135deg, rgba(25,118,210,0.08) 0%, rgba(13,71,161,0.08) 100%);
            border-radius: 12px;
            margin: 15px 0 10px 0;
            border: 2px solid rgba(25,118,210,0.15);
        }
        
        .section-title {
            color: #1565C0;
            margin: 0;
            font-size: 1.3rem;
            font-weight: 700;
            letter-spacing: -0.3px;
        }
        </style>
    """, unsafe_allow_html=True)


# ========================== MAIN APPLICATION ==========================
def main():
    load_custom_css()
    
    # Language selector in sidebar
    with st.sidebar:
        lang_col1, lang_col2 = st.columns(2)
        with lang_col1:
            if st.button("Tiếng Việt", use_container_width=True, 
                        type="primary" if st.session_state.language == 'vi' else "secondary"):
                st.session_state.language = 'vi'
                st.rerun()
        with lang_col2:
            if st.button("English", use_container_width=True,
                        type="primary" if st.session_state.language == 'en' else "secondary"):
                st.session_state.language = 'en'
                st.rerun()
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    
    # Header - compact with translation
    st.markdown(f"""
        <div style='text-align: center; margin-bottom: 1rem;'>
            <h1 style='
                background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.2rem;
                font-weight: 900;
                margin-bottom: 0.3rem;
                letter-spacing: -1px;
            '>⚕ {t('title')}</h1>
            <p style='color: #1565C0; font-size: 0.95rem; font-weight: 500;'>
                {t('subtitle')}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # System Status Card
        status_bg = '#4CAF50' if model_loaded else '#F44336'
        status_text = ('✓ ' + t('ready')) if model_loaded else '✗ Error'
        
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
                padding: 25px;
                border-radius: 16px;
                color: white;
                margin-bottom: 25px;
                box-shadow: 0 8px 24px rgba(25,118,210,0.35);
            '>
                <h2 style='margin: 0 0 18px 0; text-align: center; font-size: 1.5rem; font-weight: 800;'>
                    ⚕ {t('system_ai')}
                </h2>
                <div style='background: rgba(255,255,255,0.15); padding: 15px; border-radius: 12px; margin-top: 15px;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 12px;'>
                        <span style='font-weight: 600;'>▣ {t('device')}</span>
                        <span style='background: rgba(255,255,255,0.25); padding: 4px 12px; border-radius: 6px; font-weight: 700;'>{DEVICE.upper()}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between;'>
                        <span style='font-weight: 600;'>▣ {t('status')}</span>
                        <span style='background: {status_bg}; padding: 4px 12px; border-radius: 6px; font-weight: 700;'>{status_text}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Quick Guide with icons
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, rgba(76,175,80,0.1) 0%, rgba(56,142,60,0.05) 100%);
                padding: 20px;
                border-radius: 12px;
                border: 2px solid rgba(76,175,80,0.3);
                margin-bottom: 20px;
            '>
                <h3 style='color: #2E7D32; margin: 0 0 18px 0; text-align: center; font-weight: 800; font-size: 1.15rem;'>
                    {t('guide_title')}
                </h3>
                <div style='color: #2E7D32; line-height: 1.9;'>
                    <div style='margin: 10px 0; display: flex; align-items: flex-start;'>
                        <span style='background: #4CAF50; color: white; padding: 4px 10px; border-radius: 50%; margin-right: 12px; font-weight: 700; font-size: 0.9rem;'>①</span>
                        <span style='font-weight: 600;'>{t('guide_step1')}</span>
                    </div>
                    <div style='margin: 10px 0; display: flex; align-items: flex-start;'>
                        <span style='background: #4CAF50; color: white; padding: 4px 10px; border-radius: 50%; margin-right: 12px; font-weight: 700; font-size: 0.9rem;'>②</span>
                        <span style='font-weight: 600;'>{t('guide_step2')}</span>
                    </div>
                    <div style='margin: 10px 0; display: flex; align-items: flex-start;'>
                        <span style='background: #4CAF50; color: white; padding: 4px 10px; border-radius: 50%; margin-right: 12px; font-weight: 700; font-size: 0.9rem;'>③</span>
                        <span style='font-weight: 600;'>{t('guide_step3')}</span>
                    </div>
                    <div style='margin: 10px 0; display: flex; align-items: flex-start;'>
                        <span style='background: #4CAF50; color: white; padding: 4px 10px; border-radius: 50%; margin-right: 12px; font-weight: 700; font-size: 0.9rem;'>④</span>
                        <span style='font-weight: 600;'>{t('guide_step4')}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Statistics card
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, rgba(255,152,0,0.1) 0%, rgba(245,124,0,0.05) 100%);
                padding: 20px;
                border-radius: 12px;
                border: 2px solid rgba(255,152,0,0.3);
                margin-bottom: 20px;
            '>
                <h3 style='color: #E65100; margin: 0 0 18px 0; text-align: center; font-weight: 800; font-size: 1.15rem;'>
                    {t('model_info')}
                </h3>
                <div style='color: #E65100; font-weight: 600; line-height: 1.9;'>
                    <div style='margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.5); border-radius: 8px;'>
                        <span style='opacity: 0.8;'>{t('model_version')}:</span> <span style='float: right; font-weight: 800;'>3.0</span>
                    </div>
                    <div style='margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.5); border-radius: 8px;'>
                        <span style='opacity: 0.8;'>{t('architecture')}:</span> <span style='float: right; font-weight: 800;'>HybridViT</span>
                    </div>
                    <div style='margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.5); border-radius: 8px;'>
                        <span style='opacity: 0.8;'>{t('dataset')}:</span> <span style='float: right; font-weight: 800;'>ISIC 2018</span>
                    </div>
                    <div style='margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.5); border-radius: 8px;'>
                        <span style='opacity: 0.8;'>{t('accuracy')}:</span> <span style='float: right; font-weight: 800;'>85%+</span>
                    </div>
                    <div style='margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.5); border-radius: 8px;'>
                        <span style='opacity: 0.8;'>{t('classes')}:</span> <span style='float: right; font-weight: 800;'>9 {'types' if st.session_state.language == 'en' else 'loại'}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Warning notice
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, rgba(244,67,54,0.1) 0%, rgba(229,57,53,0.05) 100%);
                padding: 18px;
                border-radius: 12px;
                border: 2px solid rgba(244,67,54,0.3);
            '>
                <h4 style='color: #C62828; margin: 0 0 12px 0; text-align: center; font-weight: 800;'>⚠ {t('important_note')}</h4>
                <p style='color: #C62828; font-size: 0.9rem; margin: 0; line-height: 1.7; font-weight: 600;'>
                    {t('note_text')}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if not model_loaded:
        st.error(f"{t('cannot_load_model')} {CHECKPOINT_PATH}")
        st.info(t('ensure_model'))
        return
    
    # File uploader - compact version with translation
    uploaded_file = st.file_uploader(
        t('upload_title') + " (JPG, PNG, JPEG)",
        type=['jpg', 'jpeg', 'png'],
        help=t('upload_help')
    )
    
    # Flowchart when no image uploaded
    if uploaded_file is None:
        st.markdown(f"""
            <h2 style='color: #1565C0; margin: 20px 0 25px 0; font-weight: 900; font-size: 1.5rem; text-align: center;'>
                {t('workflow_title')}
            </h2>
        """, unsafe_allow_html=True)
        
        # Horizontal flowchart
        cols = st.columns([1, 0.3, 1, 0.3, 1, 0.3, 1, 0.3, 1])
        
        with cols[0]:
            st.markdown(f"""
                <div style='
                    background: white;
                    padding: 12px 10px;
                    border-radius: 10px;
                    border-top: 4px solid #1976D2;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                    text-align: center;
                '>
                    <h4 style='color: #1565C0; margin: 0; font-size: 0.95rem; font-weight: 800;'>{t('flow_step1')}</h4>
                </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown("<div style='text-align: center; padding-top: 10px; color: #1976D2; font-size: 1.3rem;'>→</div>", unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown(f"""
                <div style='
                    background: white;
                    padding: 12px 10px;
                    border-radius: 10px;
                    border-top: 4px solid #1976D2;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                    text-align: center;
                '>
                    <h4 style='color: #1565C0; margin: 0; font-size: 0.95rem; font-weight: 800;'>{t('flow_step2')}</h4>
                </div>
            """, unsafe_allow_html=True)
        
        with cols[3]:
            st.markdown("<div style='text-align: center; padding-top: 10px; color: #1976D2; font-size: 1.3rem;'>→</div>", unsafe_allow_html=True)
        
        with cols[4]:
            st.markdown(f"""
                <div style='
                    background: white;
                    padding: 12px 10px;
                    border-radius: 10px;
                    border-top: 4px solid #1976D2;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                    text-align: center;
                '>
                    <h4 style='color: #1565C0; margin: 0; font-size: 0.95rem; font-weight: 800;'>{t('flow_step3')}</h4>
                </div>
            """, unsafe_allow_html=True)
        
        with cols[5]:
            st.markdown("<div style='text-align: center; padding-top: 10px; color: #1976D2; font-size: 1.3rem;'>→</div>", unsafe_allow_html=True)
        
        with cols[6]:
            st.markdown(f"""
                <div style='
                    background: white;
                    padding: 12px 10px;
                    border-radius: 10px;
                    border-top: 4px solid #1976D2;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                    text-align: center;
                '>
                    <h4 style='color: #1565C0; margin: 0; font-size: 0.95rem; font-weight: 800;'>{t('flow_step4')}</h4>
                </div>
            """, unsafe_allow_html=True)
        
        with cols[7]:
            st.markdown("<div style='text-align: center; padding-top: 10px; color: #FF9800; font-size: 1.3rem;'>→</div>", unsafe_allow_html=True)
        
        with cols[8]:
            st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, rgba(255,193,7,0.15) 0%, rgba(255,152,0,0.1) 100%);
                    padding: 12px 10px;
                    border-radius: 10px;
                    border-top: 4px solid #FF9800;
                    box-shadow: 0 2px 6px rgba(255,152,0,0.2);
                    text-align: center;
                '>
                    <h4 style='color: #E65100; margin: 0; font-size: 0.95rem; font-weight: 800;'>{t('flow_step5')}</h4>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, rgba(255,193,7,0.1) 0%, rgba(255,152,0,0.05) 100%);
                padding: 25px;
                border-radius: 14px;
                border: 2px solid rgba(255,152,0,0.4);
                margin: 20px 0;
            '>
                <h3 style='color: #E65100; margin: 0 0 18px 0; text-align: center; font-weight: 800; font-size: 1.3rem;'>
                    {t('diseases_title')}
                </h3>
                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 15px;'>
                    <div style='background: white; padding: 12px; border-radius: 8px; text-align: center; border: 2px solid #E3F2FD;'>
                        <p style='margin: 0; color: #1565C0; font-weight: 700; font-size: 0.9rem;'>Actinic Keratosis</p>
                    </div>
                    <div style='background: white; padding: 12px; border-radius: 8px; text-align: center; border: 2px solid #E3F2FD;'>
                        <p style='margin: 0; color: #1565C0; font-weight: 700; font-size: 0.9rem;'>Basal Cell Carcinoma</p>
                    </div>
                    <div style='background: white; padding: 12px; border-radius: 8px; text-align: center; border: 2px solid #E3F2FD;'>
                        <p style='margin: 0; color: #1565C0; font-weight: 700; font-size: 0.9rem;'>Dermatofibroma</p>
                    </div>
                    <div style='background: white; padding: 12px; border-radius: 8px; text-align: center; border: 2px solid #E3F2FD;'>
                        <p style='margin: 0; color: #1565C0; font-weight: 700; font-size: 0.9rem;'>Melanoma</p>
                    </div>
                    <div style='background: white; padding: 12px; border-radius: 8px; text-align: center; border: 2px solid #E3F2FD;'>
                        <p style='margin: 0; color: #1565C0; font-weight: 700; font-size: 0.9rem;'>Nevus</p>
                    </div>
                    <div style='background: white; padding: 12px; border-radius: 8px; text-align: center; border: 2px solid #E3F2FD;'>
                        <p style='margin: 0; color: #1565C0; font-weight: 700; font-size: 0.9rem;'>Pigmented Keratosis</p>
                    </div>
                    <div style='background: white; padding: 12px; border-radius: 8px; text-align: center; border: 2px solid #E3F2FD;'>
                        <p style='margin: 0; color: #1565C0; font-weight: 700; font-size: 0.9rem;'>Seborrheic Keratosis</p>
                    </div>
                    <div style='background: white; padding: 12px; border-radius: 8px; text-align: center; border: 2px solid #E3F2FD;'>
                        <p style='margin: 0; color: #1565C0; font-weight: 700; font-size: 0.9rem;'>Squamous Cell Carcinoma</p>
                    </div>
                    <div style='background: white; padding: 12px; border-radius: 8px; text-align: center; border: 2px solid #E3F2FD;'>
                        <p style='margin: 0; color: #1565C0; font-weight: 700; font-size: 0.9rem;'>Vascular Lesion</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Analysis section - compact
        st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            st.markdown("<h3 style='color: #1565C0; text-align: center; margin-bottom: 1rem;'>Ảnh gốc</h3>", unsafe_allow_html=True)
            st.image(image, width='stretch')
        
        with col2:
            with st.spinner("Đang phân tích bằng AI..."):
                pred_class, pred_class_vi, confidence, probs = predict(image)
            
            if pred_class:
                info = CLASS_INFO[pred_class]
                
                st.markdown("<h3 style='color: #1565C0; text-align: center; margin-bottom: 1rem;'>Kết quả chẩn đoán</h3>", unsafe_allow_html=True)
                
                # Result card
                st.markdown(
                    f"""
                    <div style='
                        background: {info['gradient']};
                        padding: 32px;
                        border-radius: 20px;
                        color: white;
                        box-shadow: 0 12px 32px rgba(25,118,210,0.3);
                        margin: 15px 0;
                        border: 3px solid rgba(255,255,255,0.3);
                    '>
                        <div style='text-align: center;'>
                            <h2 style='margin: 0 0 12px 0; font-size: 2.2rem; font-weight: 900;'>{pred_class_vi}</h2>
                            <p style='margin: 0 0 20px 0; opacity: 0.95; font-size: 1rem; font-weight: 500;'>{pred_class}</p>
                            <div style='background: rgba(255,255,255,0.25); padding: 20px; border-radius: 14px;'>
                                <div style='font-size: 0.95rem; opacity: 0.95; margin-bottom: 10px; font-weight: 600; letter-spacing: 1.5px;'>ĐỘ TIN CẬY AI</div>
                                <div style='font-size: 3.5rem; font-weight: 900;'>{confidence*100:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Risk badge
                risk_colors = {
                    'Rất thấp': ('#4CAF50', 'rgba(76,175,80,0.1)'),
                    'Thấp': ('#8BC34A', 'rgba(139,195,74,0.1)'),
                    'Thấp-Trung bình': ('#FFC107', 'rgba(255,193,7,0.1)'),
                    'Trung bình': ('#FF9800', 'rgba(255,152,0,0.1)'),
                    'Cao': ('#F44336', 'rgba(244,67,54,0.1)')
                }
                risk_color, risk_bg = risk_colors.get(info['risk'], ('#9E9E9E', 'rgba(158,158,158,0.1)'))
                
                st.markdown(
                    f"""
                    <div style='
                        background: {risk_bg};
                        border: 2px solid {risk_color};
                        color: {risk_color};
                        padding: 10px 16px;
                        border-radius: 10px;
                        text-align: center;
                        font-weight: 700;
                        font-size: 0.95rem;
                        margin: 8px 0 5px 0;
                    '>
                        ⚠ Nguy hiểm: {info['risk'].upper()}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Detailed analysis - compact header
        st.markdown("""<div style='margin: 20px 0 10px 0;'><h3 style='color: #1565C0; text-align: center; font-weight: 700;'>Phân tích chi tiết</h3></div>""", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Tất cả các loại", "Top 5 dự đoán", "Thông tin bệnh"])
        
        with tab1:
            fig_bar = plot_probabilities_chart(probs, CLASS_NAMES_VI)
            st.plotly_chart(fig_bar, width='stretch')
        
        with tab2:
            col_t1, col_t2 = st.columns([3, 2], gap="large")
            
            with col_t1:
                fig_donut = plot_top_predictions(probs, CLASS_NAMES_VI)
                st.plotly_chart(fig_donut, width='stretch')
            
            with col_t2:
                st.markdown("""
                    <h3 style='
                        text-align: center;
                        color: white;
                        background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
                        padding: 14px;
                        border-radius: 12px;
                        margin-bottom: 20px;
                        font-weight: 800;
                    '>BẢNG XẾP HẠNG</h3>
                """, unsafe_allow_html=True)
                
                top5_idx = np.argsort(probs)[::-1][:5]
                rank_colors = ['#0D47A1', '#1565C0', '#1976D2', '#1E88E5', '#42A5F5']
                
                for rank, idx in enumerate(top5_idx, 1):
                    st.markdown(
                        f"""
                        <div style='
                            background: linear-gradient(135deg, rgba(25,118,210,0.1) 0%, rgba(13,71,161,0.05) 100%);
                            border-radius: 14px;
                            padding: 16px 18px;
                            margin: 12px 0;
                            border-left: 5px solid {rank_colors[rank-1]};
                            box-shadow: 0 4px 10px rgba(25,118,210,0.15);
                        '>
                            <div style='display: flex; align-items: center; justify-content: space-between;'>
                                <div style='display: flex; align-items: center; gap: 12px;'>
                                    <span style='
                                        background: {rank_colors[rank-1]};
                                        color: white;
                                        font-weight: 900;
                                        font-size: 1.4rem;
                                        width: 36px;
                                        height: 36px;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        border-radius: 50%;
                                    '>{rank}</span>
                                    <span style='font-weight: 700; color: #0D47A1; font-size: 1rem;'>{CLASS_NAMES_VI[idx]}</span>
                                </div>
                                <span style='
                                    font-weight: 900;
                                    color: {rank_colors[rank-1]};
                                    font-size: 1.2rem;
                                    background: white;
                                    padding: 6px 16px;
                                    border-radius: 10px;
                                    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                                '>{probs[idx]*100:.1f}%</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        with tab3:
            st.markdown(f"""
                <h3 style='
                    text-align: center;
                    color: white;
                    background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
                    padding: 16px;
                    border-radius: 14px;
                    margin-bottom: 24px;
                    font-weight: 800;
                    font-size: 1.4rem;
                '>THÔNG TIN VỀ {pred_class_vi.upper()}</h3>
            """, unsafe_allow_html=True)
            
            col_i1, col_i2 = st.columns(2, gap="large")
            
            with col_i1:
                st.markdown("""
                    <div class='metric-card'>
                        <h4 style='color: #1565C0; margin: 0 0 12px 0; font-weight: 800;'>MÔ TẢ CHI TIẾT</h4>
                    </div>
                """, unsafe_allow_html=True)
                st.info(info['description'])
                
            with col_i2:
                st.markdown("""
                    <div class='metric-card'>
                        <h4 style='color: #1565C0; margin: 0 0 12px 0; font-weight: 800;'>PHƯƠNG PHÁP ĐIỀU TRỊ</h4>
                    </div>
                """, unsafe_allow_html=True)
                st.success(info['treatment'])
            
            st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, rgba(25,118,210,0.1) 0%, rgba(13,71,161,0.05) 100%);
                    padding: 20px;
                    border-radius: 14px;
                    border: 3px solid #1976D2;
                    text-align: center;
                    margin-top: 20px;
                '>
                    <span style='color: #0D47A1; font-weight: 800; font-size: 1.15rem;'>MỨC ĐỘ NGUY HIỂM:</span>
                    <span style='
                        background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
                        color: white;
                        padding: 8px 24px;
                        border-radius: 10px;
                        margin-left: 12px;
                        font-weight: 900;
                        font-size: 1.15rem;
                    '>{info['risk'].upper()}</span>
                </div>
            """, unsafe_allow_html=True)

        # Gemini care advice (safety-first, non-prescriptive)
        st.markdown("<div style='margin: 25px 0 10px 0;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, rgba(25,118,210,0.06) 0%, rgba(13,71,161,0.03) 100%);
                border: 2px solid rgba(25,118,210,0.15);
                border-radius: 16px;
                padding: 18px 20px;
                box-shadow: 0 6px 16px rgba(25,118,210,0.12);
            '>
                <h3 style='color: #0D47A1; margin: 0 0 10px 0; font-weight: 850;'>{t('advice_title')}</h3>
                <p style='color: #1565C0; margin: 0 0 12px 0; font-weight: 600;'>{t('advice_desc')}</p>
            </div>
        """, unsafe_allow_html=True)

        user_notes = st.text_area(t('advice_textarea'), height=90)

        advice_placeholder = st.empty()
        if not GEMINI_API_KEY:
            advice_placeholder.warning(t('advice_missing_key'))
        else:
            if st.button(t('advice_btn'), use_container_width=True):
                with st.spinner(t('advice_loading')):
                    advice_text, advice_err = generate_gemini_advice(
                        pred_class,
                        pred_class_vi,
                        info['risk'],
                        confidence,
                        user_notes,
                        st.session_state.get('language', 'vi')
                    )
                if advice_text:
                    advice_placeholder.markdown(
                        f"""
                        <div style='background: white; border: 2px solid rgba(25,118,210,0.2); border-radius: 14px; padding: 16px; box-shadow: 0 6px 16px rgba(0,0,0,0.08); margin-top: 12px;'>
                            <h4 style='color: #0D47A1; margin: 0 0 10px 0; font-weight: 800;'>{t('advice_result_title')}</h4>
                            <div style='color: #0D47A1; line-height: 1.6; font-weight: 600;'>{advice_text}</div>
                            <p style='color: #C62828; font-size: 0.9rem; margin-top: 12px; font-weight: 700;'>⚠ {t('advice_disclaimer')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    advice_placeholder.error(t('advice_error') + (f" ({advice_err})" if advice_err else ""))
        
        # Medical disclaimer
        st.markdown("---")
        st.markdown("""
            <div style='
                background: linear-gradient(135deg, rgba(255,152,0,0.1) 0%, rgba(255,193,7,0.1) 100%);
                border: 3px solid #FF9800;
                border-radius: 18px;
                padding: 28px;
                margin: 30px 0;
                box-shadow: 0 6px 20px rgba(255,152,0,0.15);
            '>
                <h3 style='color: #E65100; margin: 0 0 18px 0; font-weight: 900; font-size: 1.4rem; text-align: center;'>
                    LƯU Ý Y TẾ QUAN TRỌNG
                </h3>
                <div style='color: #E65100; font-size: 1.05rem; line-height: 1.8;'>
                    <p style='font-weight: 600; margin-bottom: 18px;'>
                        Ứng dụng này chỉ mang tính chất tham khảo và hỗ trợ, <strong>KHÔNG thay thế</strong> cho chẩn đoán y khoa chuyên nghiệp.
                    </p>
                    <div style='background: rgba(255,255,255,0.7); padding: 18px; border-radius: 12px; margin-top: 12px;'>
                        <p style='margin: 10px 0; font-weight: 600;'><strong>▸</strong> Luôn tham khảo ý kiến bác sĩ da liễu có chứng chỉ hành nghề</p>
                        <p style='margin: 10px 0; font-weight: 600;'><strong>▸</strong> Kết quả AI chỉ là công cụ hỗ trợ, không phải chẩn đoán cuối cùng</p>
                        <p style='margin: 10px 0; font-weight: 600;'><strong>▸</strong> Hãy khám sức khỏe định kỳ và theo dõi sự thay đổi của da</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


# ========================== RUN APP ==========================
if __name__ == "__main__":
    main()
