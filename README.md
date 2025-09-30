# 🎭 AI Emotion Detection System

A comprehensive real-time emotion detection application using deep learning models with both CNN and DNN architectures. Features a modern GUI interface built with tkinter and supports multiple model backends including Hugging Face transformers.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎬 Demo Videos

Experience the AI Emotion Detection System in action! We've prepared demonstration videos showcasing both model architectures:

### 🧠 CNN Model Demo
**File:** `cnn.mp4` (29MB)

<video controls src="https://github.com/sahanruwantha/computer-vision-assignment/blob/main/cnn.mp4" title="https://github.com/sahanruwantha/computer-vision-assignment/cnn.mp4"></video>

The Convolutional Neural Network model demonstration showcases:
- High-accuracy emotion detection in real-time
- Robust performance across different lighting conditions
- Detailed emotion classification with confidence scores
- Smooth real-time processing with minimal latency

### 🔄 DNN Model Demo  
**File:** `dnn.mp4` (40MB)

<video controls src="https://github.com/sahanruwantha/computer-vision-assignment/blob/main/dnn.mp4" title="https://github.com/sahanruwantha/computer-vision-assignment/dnn.mp4"></video>

The Deep Neural Network model demonstration highlights:
- Fast inference and lightweight processing
- Efficient emotion detection suitable for resource-constrained environments
- Real-time performance optimization
- Comparative analysis with CNN model results

> **📹 Video Access:** You can download and view the demo videos directly from the repository files above, The videos demonstrate the complete workflow from face detection to emotion classification using the modern GUI interface.

## 🌟 Features

- **Real-time emotion detection** from webcam feed
- **Multiple model support**: CNN, DNN, and Hugging Face transformers
- **Beautiful modern GUI** with dark theme
- **7 emotion categories**: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- **Live statistics** and performance metrics
- **Screenshot functionality** to save detection results
- **Model switching** without restarting the application
- **Camera selection** support for multiple cameras

## 📋 Requirements

- Python 3.10 or higher
- Webcam/Camera
- CUDA-capable GPU (optional, CPU fallback available)

## 🚀 Quick Start

### Option 1: Using uv (Recommended)

```bash
# Clone or download the project
cd computer-vision

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the GUI application
uv run python emotion_gui.py

# Or run the command-line version
uv run python main.py
```

### Option 2: Using pip

```bash
# Clone or download the project
cd computer-vision

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the GUI application
python emotion_gui.py

# Or run the command-line version
python main.py
```

## 📦 Installation

### Dependencies

The project requires the following main dependencies:

- **OpenCV** (`opencv-python>=4.12.0`) - Computer vision and camera handling
- **TensorFlow** (`tensorflow>=2.20.0`) - Deep learning framework
- **Pillow** (`pillow>=10.0.0`) - Image processing
- **NumPy** (`numpy>=1.24.0`) - Numerical computations
- **Transformers** (`transformers>=4.56.2`) - Hugging Face models (optional)
- **PyTorch** (`torch>=2.8.0`) - Alternative deep learning framework
- **tkinter** - GUI framework (usually included with Python)

### Model Files

The application expects the following pre-trained models:
- `emotion_cnn_model.h5` - Convolutional Neural Network model
- `emotion_dnn_model.h5` - Deep Neural Network model
- `facial_recognition_model.keras` - Face detection model (optional)

If models are not present, you can train them using `train.py` script.

## 🎯 Usage

### GUI Application

1. **Launch the application:**
   ```bash
   python emotion_gui.py
   ```

2. **Select a model** from the dropdown (CNN or DNN)

3. **Set camera index** (usually 0 for default camera)

4. **Click "Start Detection"** to begin real-time emotion detection

5. **Use controls:**
   - 🚀 **Start Detection** - Begin emotion detection
   - ⏹️ **Stop Detection** - Stop the detection process
   - 📷 **Screenshot** - Save current frame with detections

### Command Line Interface

```bash
# Basic usage with default CNN model
python main.py

# Use DNN model
python main.py --model dnn

# Use Hugging Face model (requires transformers)
python main.py --model huggingface

# Specify custom model path
python main.py --model-path /path/to/your/model.h5 --model-type cnn

# Change camera index
python main.py --camera 1
```

### Command Line Arguments

- `--model-path`: Path to the emotion detection model
- `--model-type`: Model type ('cnn', 'dnn', or 'huggingface')
- `--camera`: Camera index (default: 0)
- `--confidence-threshold`: Minimum confidence for predictions (default: 0.5)

## 🏗️ Project Structure

```
computer-vision/
│
├── 📊 Models
│   ├── emotion_cnn_model.h5         # Pre-trained CNN model
│   ├── emotion_dnn_model.h5         # Pre-trained DNN model
│   └── facial_recognition_model.keras # Face detection model
│
├── 🖥️ Applications
│   ├── emotion_gui.py               # GUI application
│   ├── main.py                      # CLI application & core logic
│   └── main_huggingface.py          # Hugging Face integration
│
├── 🔧 Training & Setup
│   ├── train.py                     # Model training script
│   ├── train_data.zip              # Training dataset
│   └── train_data/                 # Extracted training data
│       ├── happy/
│       ├── sad/
│       └── ...
│
├── 📝 Configuration
│   ├── requirements.txt             # pip dependencies
│   ├── requirements_hf.txt          # Hugging Face requirements
│   ├── pyproject.toml              # Project configuration
│   ├── uv.lock                     # uv lock file
│   └── README.md                   # This file
│
└── 📋 Documentation
    └── report.md                   # Project report
```

## 🎨 GUI Features

### Main Interface
- **Dark theme** with modern styling
- **Real-time video feed** with emotion overlays
- **Model information panel** with current model details
- **Detection statistics** including FPS and face count
- **Emotion color legend** for easy identification

### Controls
- **Model Selection**: Switch between CNN and DNN models
- **Camera Selection**: Choose camera source
- **Start/Stop**: Control detection process
- **Screenshot**: Save current frame with detections

## 🧠 Model Information

### CNN Model
- **Architecture**: Convolutional layers with pooling and dropout
- **Input**: 48x48 grayscale images
- **Output**: 7 emotion classes
- **Performance**: Higher accuracy, more computationally intensive

### DNN Model
- **Architecture**: Fully connected layers with dropout
- **Input**: 48x48 flattened grayscale images
- **Output**: 7 emotion classes
- **Performance**: Faster inference, lower accuracy

### Hugging Face Models
- **Pre-trained transformers** for emotion detection
- **Various architectures** available
- **Easy to switch** between different models

## 📊 Supported Emotions

The system can detect the following 7 emotional states:

1. 😊 **Happy** (Green)
2. 😢 **Sad** (Red)
3. 😠 **Angry** (Blue)
4. 😲 **Surprise** (Cyan)
5. 😨 **Fear** (Purple)
6. 🤢 **Disgust** (Dark Green)
7. 😐 **Neutral** (Gray)

## 🔧 Training Custom Models

To train your own models:

1. **Prepare your dataset** in the following structure:
   ```
   train_data/
   ├── happy/
   ├── sad/
   ├── angry/
   ├── surprise/
   ├── fear/
   ├── disgust/
   └── neutral/
   ```

2. **Run the training script:**
   ```bash
   python train.py
   ```

3. **Models will be saved as:**
   - `emotion_cnn_model.h5`
   - `emotion_dnn_model.h5`

## ⚡ Performance Optimization

### CPU Optimization
- Models are configured to run on CPU by default
- GPU usage is disabled to avoid CUDA compatibility issues
- Optimized for real-time performance on standard hardware

### Memory Management
- Efficient image preprocessing and memory cleanup
- Background thread processing to maintain UI responsiveness
- Automatic frame rate limiting to prevent resource overuse

## 🔍 Troubleshooting

### Common Issues

1. **Camera not detected:**
   - Check camera permissions
   - Try different camera indices (0, 1, 2, etc.)
   - Ensure no other applications are using the camera

2. **Model loading errors:**
   - Verify model files exist in the project directory
   - Check TensorFlow/Keras compatibility
   - Try reinstalling dependencies

3. **Low performance:**
   - Reduce video resolution
   - Lower frame rate in detection loop
   - Use DNN model instead of CNN for faster processing

4. **GUI issues:**
   - Ensure tkinter is properly installed
   - Check display settings and resolution
   - Try running in different Python environments

### Error Messages

- **"Model file not found"**: Download or train the required model files
- **"Cannot open camera"**: Check camera connection and permissions
- **"ImportError"**: Install missing dependencies with pip or uv

## 🙏 Acknowledgments

- **FER2013 Dataset** for emotion recognition training data
- **OpenCV** for computer vision capabilities
- **TensorFlow/Keras** for deep learning framework
- **Hugging Face** for pre-trained transformer models
- **tkinter** for GUI framework