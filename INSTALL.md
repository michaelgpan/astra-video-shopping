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
