#!/bin/bash

# Script to setup Git LFS for model files
# Usage: bash setup_git_lfs.sh

echo "🚀 Setting up Git LFS for Skin Cancer VIT Project"
echo "=================================================="

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS is not installed"
    echo ""
    echo "Please install Git LFS first:"
    echo "  macOS:   brew install git-lfs"
    echo "  Ubuntu:  sudo apt-get install git-lfs"
    echo "  Windows: Download from https://git-lfs.github.com/"
    exit 1
fi

echo "✅ Git LFS found: $(git-lfs version)"

# Initialize Git LFS in the repo
echo ""
echo "📦 Initializing Git LFS..."
git lfs install
if [ $? -eq 0 ]; then
    echo "✅ Git LFS initialized"
else
    echo "❌ Failed to initialize Git LFS"
    exit 1
fi

# Check if .gitattributes already has LFS rules
if grep -q "filter=lfs" .gitattributes; then
    echo "✅ Git LFS rules already exist in .gitattributes"
else
    echo "⚠️  Git LFS rules not found in .gitattributes"
    echo "Please ensure .gitattributes contains LFS rules"
fi

# Show tracked files
echo ""
echo "📋 Currently tracked LFS files:"
git lfs ls-files

# Check if model files exist
echo ""
echo "🔍 Checking for model files..."
if [ -f "best_model_CNN_CBAM_ViT.pt" ]; then
    SIZE=$(du -h best_model_CNN_CBAM_ViT.pt | cut -f1)
    echo "✅ Found: best_model_CNN_CBAM_ViT.pt ($SIZE)"
else
    echo "❌ Not found: best_model_CNN_CBAM_ViT.pt"
fi

if [ -f "model/best_model_CNN_CBAM_ViT.pt" ]; then
    SIZE=$(du -h model/best_model_CNN_CBAM_ViT.pt | cut -f1)
    echo "✅ Found: model/best_model_CNN_CBAM_ViT.pt ($SIZE)"
else
    echo "❌ Not found: model/best_model_CNN_CBAM_ViT.pt"
fi

# Instructions
echo ""
echo "📝 Next steps:"
echo "=============="
echo "1. Stage the model files:"
echo "   git add best_model_CNN_CBAM_ViT.pt"
echo "   git add model/best_model_CNN_CBAM_ViT.pt"
echo "   git add .gitattributes"
echo ""
echo "2. Commit the changes:"
echo "   git commit -m 'Add model files with Git LFS'"
echo ""
echo "3. Push to GitHub:"
echo "   git push origin main"
echo ""
echo "4. Verify on GitHub:"
echo "   - Model files should show 'Stored with Git LFS' label"
echo "   - Check repository size didn't increase much"
echo ""
echo "5. Deploy on Streamlit Cloud:"
echo "   - Git LFS is automatically supported"
echo "   - Model will be downloaded during deployment"
echo ""
echo "✨ Done! Your model files are ready for Git LFS"
