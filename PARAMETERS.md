# Tài liệu Thông số Mô hình CNN-CBAM-ViT

## 1. Cấu hình Cơ bản

| Thông số | Giá trị | Mô tả |
|----------|---------|-------|
| `SEED` | 42 | Giá trị khởi tạo ngẫu nhiên để tái lập kết quả |
| `device` | cuda/cpu | Thiết bị tính toán (GPU nếu có, CPU nếu không) |
| `NUM_CLASSES` | 9 | Số lượng loại ung thư da cần phân loại |

## 2. Thông số Huấn luyện

| Thông số | Giá trị | Mô tả |
|----------|---------|-------|
| `BATCH_SIZE` | 32 | Số lượng ảnh xử lý đồng thời mỗi batch |
| `NUM_EPOCHS` | 50 | Số lượng epoch huấn luyện tối đa |
| `LEARNING_RATE` | 3e-4 (0.0003) | Tốc độ học ban đầu cho optimizer |
| `WEIGHT_DECAY` | 1e-4 (0.0001) | Hệ số regularization L2 (phòng overfitting) |
| `PATIENCE` | 6 | Số epoch chờ trước khi early stopping |

## 3. Xử lý Dữ liệu

| Thông số | Giá trị | Mô tả |
|----------|---------|-------|
| `OVERSAMPLE_FACTOR` | 5 | Hệ số tăng mẫu cho class thiểu số |
| `NUM_WORKERS` | 2 | Số luồng xử lý dữ liệu song song |
| `PIN_MEMORY` | True (GPU) | Tối ưu truyền dữ liệu lên GPU |
| Train/Val Split | 70/30 | Tỷ lệ chia tập train và validation |

### Augmentation (RandomAugmentationPerImage)
- **Resize**: 224×224 pixels
- **RandomHorizontalFlip**: p=0.5
- **RandomVerticalFlip**: p=0.5
- **RandomRotation**: ±40°
- **ColorJitter**: brightness=0.2, contrast=0.2, saturation=0.2
- **Normalize**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## 4. Optimizer & Scheduler

| Thông số | Giá trị | Mô tả |
|----------|---------|-------|
| Optimizer | AdamW | Adaptive optimizer với weight decay |
| Scheduler | CosineAnnealingLR | Giảm learning rate theo hàm cosine |
| `T_MAX` | 20 | Chu kỳ cosine annealing (epochs) |

## 5. Loss Function

| Thông số | Giá trị | Mô tả |
|----------|---------|-------|
| `USE_FOCAL` | True | Sử dụng Focal Loss thay vì CrossEntropy |
| `gamma` | 2.0 | Tham số focusing (tập trung học mẫu khó) |
| `class_weights` | Auto | Tính tự động từ phân bố class (balanced) |

**Focal Loss**: $FL(p_t) = -(1-p_t)^\gamma \log(p_t)$

## 6. Kiến trúc Mô hình

### CNN Extractor (CNNExtractorCBAM)
```
Input: 224×224×3
├─ Conv1: 3→32 channels, MaxPool → 112×112×32
├─ Conv2: 32→64 channels, MaxPool → 56×56×64
├─ Conv3: 64→128 channels, MaxPool → 28×28×128
└─ CBAM: Channel + Spatial Attention
```

### CBAM (Convolutional Block Attention Module)

| Component | Thông số | Mô tả |
|-----------|----------|-------|
| **Channel Attention** | reduction=16 | MLP với tỷ lệ giảm kênh 1/16 |
| **Spatial Attention** | kernel_size=7 | Conv 7×7 cho attention không gian |

### Patch Embedding
- **Input channels**: 128 (từ CNN)
- **Patch size**: 2×2
- **Embedding dimension**: 768

### Vision Transformer (ViT-Base)
- **Backbone**: vit_base_patch16_224 (pretrained)
- **Embed dimension**: 768
- **Số lượng heads**: 12
- **Số lượng blocks**: 12
- **Custom patch_embed**: Thay thế bằng CNN+CBAM

### Classifier
- **Input**: 768 (CLS token từ ViT)
- **Output**: 9 classes

## 7. Checkpoint & Logging

| Thư mục | Mục đích |
|---------|----------|
| `logs/` | File CSV ghi lịch sử huấn luyện |
| `checkpoints/` | Lưu model mỗi 10 epochs |
| `best_model/` | Lưu model có F1-score cao nhất |
| `results/` | Lưu biểu đồ và kết quả đánh giá |

### Metrics được theo dõi
- Train Loss, Train Accuracy
- Validation Loss, Validation Accuracy
- **Macro F1-score** (metric chính cho early stopping)
- Confusion Matrix
- Classification Report (precision, recall, F1 cho từng class)

## 8. Các Class Phân loại

| ID | Class Name | Tên Tiếng Việt |
|----|-----------|----------------|
| 0 | actinic keratosis | Sừng hóa quang hóa |
| 1 | basal cell carcinoma | Ung thư tế bào đáy |
| 2 | dermatofibroma | U xơ da |
| 3 | melanoma | Ung thư hắc tố |
| 4 | nevus | Nốt ruồi |
| 5 | pigmented benign keratosis | Sừng hóa lành tính có sắc tố |
| 6 | seborrheic keratosis | Sừng hóa tiết nhớn |
| 7 | squamous cell carcinoma | Ung thư tế bào vảy |
| 8 | vascular lesion | Tổn thương mạch máu |

## 9. Xử lý Imbalanced Data

### Chiến lược áp dụng:
1. **Oversampling**: Tăng gấp 5 lần với trọng số class
2. **Class Weights**: Tính tự động từ phân bố (sklearn.compute_class_weight)
3. **Focal Loss**: Tăng trọng số cho mẫu khó phân loại (γ=2.0)
4. **Stratified Split**: Đảm bảo tỷ lệ class trong train/val

## 10. Hardware & Performance

### Khuyến nghị:
- **GPU**: CUDA-compatible (>= 8GB VRAM)
- **RAM**: >= 16GB
- **Storage**: >= 10GB cho dataset + checkpoints

### Thời gian ước tính:
- **1 epoch**: ~5-10 phút (tùy GPU)
- **Full training** (50 epochs): ~4-8 giờ
- **Early stopping**: Thường dừng sau 15-25 epochs

## 11. Tham khảo

- ViT Paper: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- CBAM Paper: [Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- Focal Loss: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- Dataset: [ISIC - International Skin Imaging Collaboration](https://www.isic-archive.com/)
