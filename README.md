# Shopping Fullscreen - Fashion Shopping Demo

A high-performance video shopping application on Synaptics Astra with real-time object detection and fashion matching.

## Features

### 🎬 Video Playback
- **4K 60FPS Performance**: Optimized Wayland fullscreen video playback on Astra
- **Smooth Playback**: Direct decode-to-display pipeline for maximum performance
- **Cross-platform**: Works on Wayland/Linux target devices

### 🎯 Object Detection
- **YOLO Integration**: Real-time person detection using Synap YOLO models
- **Person-Only Filtering**: Only detects and processes human subjects (class 0)
- **Bounding Box Navigation**: Navigate between detected persons with arrow keys

### 👗 Fashion Matching
- **FashionCLIP AI**: Advanced fashion similarity matching using CLIP embeddings
- **Pre-loaded Models**: Fast first-time matching with pre-initialized models
- **12 Similar Items**: Displays top 12 fashion matches in a 3x4 grid
- **Real-time Results**: Background processing with instant UI updates

### 🖥️ User Interface
- **Fullscreen Video**: Immersive 4K video experience
- **Qt Overlay**: Clean pause interface with detection results
- **Control Buttons**: Play/Pause, Previous/Next, Exit controls
- **Keyboard Support**: Space (pause/resume), Arrow keys (navigate), Q (quit)

## Usage

### 1. Create Virtual Environment
```bash
# Create virtual environment with Python 3.10+
./setup_shopping.sh
source venv/bin/activate 
```

### 2. Install Python Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install below packages manually in case of setup failure 
pip install torch transformers pillow
pip install https://github.com/synaptics-synap/synap-python/releases/download/v0.0.4-preview/synap_python-0.0.4-cp310-cp310-manylinux_2_35_aarch64.whl
```

### Running the Application
```bash
export XDG_RUNTIME_DIR=/var/run/user/0
export WAYLAND_DISPLAY=wayland-1
python shopping_fullscreen.py [--video VIDEO_PATH]
```

**Arguments:**
- `--video` / `-v`: Path to video file (default: `samples/clip.mp4`)


### Controls
- **Space**: Pause/Resume video
- **Left Arrow**: Navigate to previous person detection
- **Right Arrow**: Navigate to next person detection  
- **Q**: Quit application
- **Mouse**: Click UI buttons for same functions

### Workflow
1. **One-time Setup**: Run `python embedding_model_prepare.py` to download models and images
2. **Startup**: Application loads pre-prepared FashionCLIP models and embeddings (fast!)
3. **Video Plays**: 4K 60FPS fullscreen video playback
4. **User Pauses**: Press Space to pause and analyze current frame
5. **Detection**: YOLO detects all persons in the frame
6. **Fashion Matching**: FashionCLIP finds similar fashion items (instant!)
7. **Navigation**: Use arrow keys to explore different person detections
8. **Resume**: Press Space to continue video playback

## Technical Architecture

### Dependencies
- **GStreamer**: Video decoding and display
- **PyQt5**: User interface and image handling
- **YOLO/Synap**: Object detection models
- **FashionCLIP**: Fashion similarity matching
- **PIL/Pillow**: Image processing

## File Structure
```
├── embedding_model_prepare.py # One-time setup script (run first!)
├── shopping_fullscreen.py     # Main fullscreen application
├── shopping.py               # Alternative Qt-embedded application
├── object_detection.py       # YOLO detection services
├── image_embedding.py        # FashionCLIP embedding system
├── requirements.txt          # Python dependencies
├── INSTALL.md               # Installation guide
├── samples/                  # Sample video files
│   ├── clip.mp4             # Default video
│   └── clip_1.mp4           # Alternative video
├── images/                   # Fashion item database (created by prepare script)
│   └── img_*.png            # Fashion reference images
├── embeddings/              # Pre-computed embeddings (created by prepare script)
│   └── embeddings.npy       # FashionCLIP embeddings
├── onnx/                    # ONNX model files (created by prepare script)
│   └── *.onnx               # Optimized inference models
└── cache/                   # Temporary files (auto-created)
```

## System Requirements

### Target Hardware
- **Linux/Wayland**: Optimized for Wayland compositor
- **RAM**: 4GB+ for model loading and 4K video
- **Storage**: ~2GB for models and fashion database

### Development Environment  
- **Python 3.10+**
- **GStreamer 1.16+** with Wayland support
- **PyQt5** with video capabilities
- **ONNX Runtime** for FashionCLIP inference

### Memory Usage
- Models: ~1GB (FashionCLIP + YOLO)
- Embeddings: ~200MB (1000 fashion items)
- Video Buffers: ~100MB (4K frames)
- Total: ~1.5GB typical usage

## Troubleshooting

### Common Issues
- **Video not displaying**: Check GStreamer Wayland support
- **Slow first match**: Verify embeddings are pre-loaded at startup  
- **Detection failures**: Ensure YOLO model path is correct
- **UI not responding**: Check Qt5 installation and permissions

### Logs
Application provides detailed logging for debugging:
```bash
python shopping_fullscreen.py 2>&1 | tee shopping.log
```

## Development

### Architecture
- **PopupWindow**: Main Qt application window and UI management
- **FullscreenVideoPlayer**: GStreamer video pipeline management
- **TinyFocusWindow**: Keyboard input handling during fullscreen
- **ObjectDetectionService**: YOLO detection integration
- **ImageEmbedding**: FashionCLIP similarity matching

---
*Optimized for fashion retail and shopping experiences with real-time AI-powered recommendations.*

