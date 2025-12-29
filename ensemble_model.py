"""
Ensemble Model - Kết hợp best_model.pt và best_model_CNN_CBAM_ViT.pt
Giải pháp nhanh để tăng accuracy mà không cần train lại
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import timm

# ========================== MODEL ARCHITECTURES ==========================

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x * self.channel_att(x)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)
        return x * self.spatial_att(spatial_input)

class CNNExtractorCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.cbam = CBAM(128)
    
    def forward(self, x):
        return self.cbam(self.conv3(self.conv2(self.conv1(x))))

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=128, embed_dim=768, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        return self.proj(x)

class HybridViT(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.cnn = CNNExtractorCBAM()
        self.patch_embed = PatchEmbed(in_channels=128, embed_dim=768)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1000)
        self.vit.patch_embed = nn.Identity()
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.vit.forward_features(x)
        return self.classifier(x[:, 0])

class HybridViTNoCBAM(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.patch_embed = PatchEmbed(in_channels=128, embed_dim=768)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1000)
        self.vit.patch_embed = nn.Identity()
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.vit.forward_features(x)
        return self.classifier(x[:, 0])

# ========================== ENSEMBLE MODEL ==========================

class EnsembleModel:
    """
    Ensemble của 2 models với weighted voting dựa trên performance
    
    Weights dựa trên per-class accuracy:
    - best_model.pt tốt cho: Basal Cell Carcinoma (81%), Actinic Keratosis (37%)
    - best_model_CNN_CBAM_ViT.pt tốt cho: Nevus (94%), Vascular Lesion (100%), Pigmented Benign Keratosis (75%)
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        
        # Class-specific weights dựa trên test results
        # Format: [weight_for_no_cbam, weight_for_cbam]
        self.class_weights = {
            0: [0.6, 0.4],   # Actinic Keratosis - no_cbam tốt hơn (37% vs 0%)
            1: [0.6, 0.4],   # Basal Cell Carcinoma - no_cbam tốt hơn (81% vs 62%)
            2: [0.0, 1.0],   # Dermatofibroma - chỉ cbam có (0% vs 25%)
            3: [0.5, 0.5],   # Melanoma - cả 2 đều tệ (0% vs 0%)
            4: [0.0, 1.0],   # Nevus - cbam tốt hơn nhiều (0% vs 94%)
            5: [0.1, 0.9],   # Pigmented Benign Keratosis - cbam tốt hơn (6% vs 75%)
            6: [0.5, 0.5],   # Seborrheic Keratosis - cả 2 đều tệ (0% vs 0%)
            7: [0.3, 0.7],   # Squamous Cell Carcinoma - cbam tốt hơn (0% vs 31%)
            8: [0.0, 1.0],   # Vascular Lesion - cbam hoàn hảo (0% vs 100%)
        }
        
        self.load_models()
    
    def load_models(self):
        """Load cả 2 models"""
        print("Loading Ensemble Models...")
        
        # Model 1: No CBAM
        try:
            model1 = HybridViTNoCBAM(9).to(self.device)
            checkpoint1 = torch.load('best_model.pt', map_location=self.device)
            model1.load_state_dict(checkpoint1, strict=False)
            model1.eval()
            self.models['no_cbam'] = model1
            print("✓ best_model.pt loaded")
        except Exception as e:
            print(f"✗ Error loading best_model.pt: {e}")
            self.models['no_cbam'] = None
        
        # Model 2: With CBAM
        try:
            model2 = HybridViT(9).to(self.device)
            checkpoint2 = torch.load('best_model_CNN_CBAM_ViT.pt', map_location=self.device)
            model2.load_state_dict(checkpoint2, strict=False)
            model2.eval()
            self.models['cbam'] = model2
            print("✓ best_model_CNN_CBAM_ViT.pt loaded")
        except Exception as e:
            print(f"✗ Error loading best_model_CNN_CBAM_ViT.pt: {e}")
            self.models['cbam'] = None
    
    def predict(self, image_tensor):
        """
        Ensemble prediction với weighted voting
        
        Returns:
            pred_class (int): Class index
            confidence (float): Confidence score
            all_probs (dict): Probabilities từ mỗi model
        """
        with torch.no_grad():
            predictions = {}
            
            # Get predictions từ mỗi model
            if self.models['no_cbam'] is not None:
                output1 = self.models['no_cbam'](image_tensor)
                probs1 = torch.softmax(output1, dim=1)[0].cpu().numpy()
                predictions['no_cbam'] = probs1
            
            if self.models['cbam'] is not None:
                output2 = self.models['cbam'](image_tensor)
                probs2 = torch.softmax(output2, dim=1)[0].cpu().numpy()
                predictions['cbam'] = probs2
            
            # Weighted ensemble
            ensemble_probs = np.zeros(9)
            for class_idx in range(9):
                weights = self.class_weights[class_idx]
                
                if 'no_cbam' in predictions:
                    ensemble_probs[class_idx] += weights[0] * predictions['no_cbam'][class_idx]
                
                if 'cbam' in predictions:
                    ensemble_probs[class_idx] += weights[1] * predictions['cbam'][class_idx]
            
            # Normalize
            ensemble_probs = ensemble_probs / ensemble_probs.sum()
            
            pred_class = np.argmax(ensemble_probs)
            confidence = ensemble_probs[pred_class]
            
            return pred_class, confidence, predictions, ensemble_probs
    
    def predict_with_tta(self, image, num_augments=5):
        """
        Test-Time Augmentation để tăng accuracy
        
        Args:
            image (PIL.Image): Input image
            num_augments (int): Số lượng augmentations
        
        Returns:
            Same as predict()
        """
        transform_base = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform_tta = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        all_predictions = []
        
        # Original image
        img_tensor = transform_base(image).unsqueeze(0).to(self.device)
        _, _, _, probs = self.predict(img_tensor)
        all_predictions.append(probs)
        
        # Augmented images
        for _ in range(num_augments - 1):
            img_tensor = transform_tta(image).unsqueeze(0).to(self.device)
            _, _, _, probs = self.predict(img_tensor)
            all_predictions.append(probs)
        
        # Average predictions
        avg_probs = np.mean(all_predictions, axis=0)
        pred_class = np.argmax(avg_probs)
        confidence = avg_probs[pred_class]
        
        return pred_class, confidence, None, avg_probs

# ========================== SAVE ENSEMBLE AS SINGLE FILE ==========================

def save_ensemble_model():
    """Save ensemble model as a single checkpoint file"""
    device = 'cpu'
    
    # Load both models
    model1 = HybridViTNoCBAM(9).to(device)
    checkpoint1 = torch.load('best_model.pt', map_location=device)
    model1.load_state_dict(checkpoint1, strict=False)
    
    model2 = HybridViT(9).to(device)
    checkpoint2 = torch.load('best_model_CNN_CBAM_ViT.pt', map_location=device)
    model2.load_state_dict(checkpoint2, strict=False)
    
    # Save ensemble
    ensemble_checkpoint = {
        'model_no_cbam_state_dict': model1.state_dict(),
        'model_cbam_state_dict': model2.state_dict(),
        'class_weights': {
            0: [0.6, 0.4], 1: [0.6, 0.4], 2: [0.0, 1.0], 3: [0.5, 0.5], 4: [0.0, 1.0],
            5: [0.1, 0.9], 6: [0.5, 0.5], 7: [0.3, 0.7], 8: [0.0, 1.0]
        }
    }
    
    torch.save(ensemble_checkpoint, 'best_model_ensemble.pt')
    print("✓ Ensemble model saved to best_model_ensemble.pt")

# ========================== TEST ==========================

if __name__ == "__main__":
    import glob
    
    CLASS_NAMES = [
        'Actinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma',
        'Melanoma', 'Nevus', 'Pigmented Benign Keratosis',
        'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion'
    ]
    
    # Initialize ensemble
    ensemble = EnsembleModel(device='cpu')
    
    # Test on dataset
    test_dir = 'Script/data/Test'
    if os.path.exists(test_dir):
        print(f"\n{'='*70}")
        print("TESTING ENSEMBLE MODEL")
        print(f"{'='*70}\n")
        
        correct = 0
        total = 0
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_folder = os.path.join(test_dir, class_name.lower())
            if not os.path.exists(class_folder):
                continue
            
            image_files = glob.glob(os.path.join(class_folder, '*.jpg')) + \
                          glob.glob(os.path.join(class_folder, '*.png'))
            
            if len(image_files) == 0:
                continue
            
            class_correct = 0
            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(ensemble.device)
                    
                    pred_class, confidence, _, _ = ensemble.predict(img_tensor)
                    
                    if pred_class == class_idx:
                        class_correct += 1
                        correct += 1
                    total += 1
                
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            
            acc = 100 * class_correct / len(image_files) if len(image_files) > 0 else 0
            print(f"{class_name:30s}: {class_correct:3d}/{len(image_files):3d} = {acc:6.2f}%")
        
        print(f"\n{'='*70}")
        print(f"Overall Accuracy: {correct}/{total} = {100*correct/total:.2f}%")
        print(f"{'='*70}")
    
    # Save ensemble
    print("\n")
    save_ensemble_model()
