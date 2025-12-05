#!/bin/bash

# Modern UI Launcher Script for Skin Cancer Detection System
# Version 2.0

echo "================================================"
echo "🚀 Starting Modern Skin Cancer Detection UI"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✓ Found virtual environment"
    source .venv/bin/activate
else
    echo "⚠️  No virtual environment found"
fi

# Check Python
if command -v python3 &> /dev/null; then
    echo "✓ Python3 found: $(python3 --version)"
else
    echo "✗ Python3 not found!"
    exit 1
fi

# Check if model exists
if [ -f "best_model.pt" ]; then
    echo "✓ Model file found: best_model.pt"
else
    echo "⚠️  Model file not found: best_model.pt"
    echo "Please ensure the model file is in the current directory"
fi

echo ""
echo "================================================"
echo "📦 Checking dependencies..."
echo "================================================"

# Check required packages
python3 -c "import streamlit" 2>/dev/null && echo "✓ streamlit" || echo "✗ streamlit (run: pip install streamlit)"
python3 -c "import torch" 2>/dev/null && echo "✓ torch" || echo "✗ torch (run: pip install torch)"
python3 -c "import plotly" 2>/dev/null && echo "✓ plotly" || echo "✗ plotly (run: pip install plotly)"
python3 -c "import timm" 2>/dev/null && echo "✓ timm" || echo "✗ timm (run: pip install timm)"

echo ""
echo "================================================"
echo "🌐 Launching application..."
echo "================================================"
echo ""
echo "The app will open in your browser at:"
echo "👉 http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Run the modern UI
streamlit run app_modern.py
