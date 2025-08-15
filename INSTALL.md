# Installation Guide - Shopping Fullscreen

## Python Virtual Environment Setup

### 1. Create Virtual Environment
```bash
# Create virtual environment with Python 3.10+
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 2. Install Python Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

### Prepare Models and Images (One-time Setup)
```bash
# Activate virtual environment
source venv/bin/activate

# Download and prepare FashionCLIP models and images (takes 10-15 minutes)
python embedding_model_prepare.py

# Check preparation status
python embedding_model_prepare.py --help
```

### Run Application
```bash
# Activate virtual environment
source venv/bin/activate

# Run with default video
python shopping_fullscreen.py

# Run with custom video
python shopping_fullscreen.py --video /path/to/your/video.mp4

# Alternative: Run Qt-embedded version
python shopping.py
```
---
*For target device deployment, ensure all system dependencies are installed and properly configured.*
