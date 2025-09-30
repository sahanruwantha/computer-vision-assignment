# 🎭 AI Emotion Detection System

A real-time emotion detection application using deep learning models with both CNN and DNN architectures. Features live webcam emotion detection with an enhanced OpenCV interface.

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
- **Multiple model support**: CNN and DNN architectures
- **Enhanced OpenCV interface** with fullscreen support
- **2 emotion categories**: Happy and Sad
- **Live statistics** including FPS and face count
- **Screenshot functionality** to save detection results
- **Fullscreen toggle** for immersive experience
- **Performance monitoring** with real-time FPS display

## 📋 Requirements

- Python 3.10 or higher
- Webcam/Camera
- TensorFlow/Keras
- OpenCV
- NumPy

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project
cd computer-vision-assignment

# Install dependencies with uv
uv add tensorflow opencv-python numpy

# Run the application
uv run python main.py
```

## 📦 Dependencies

The project requires the following main dependencies:

- **OpenCV** (`opencv-python`) - Computer vision and camera handling
- **TensorFlow** (`tensorflow`) - Deep learning framework
- **NumPy** (`numpy`) - Numerical computations

### Model Files

The application expects the following pre-trained models:
- `emotion_cnn_model.h5` - Convolutional Neural Network model
- `emotion_dnn_model.h5` - Deep Neural Network model

If models are not present, you can train them using the `train.ipynb` notebook.

## 🎯 Usage

### Command Line Interface

```bash
# Use CNN model (default)
uv run python main.py
uv run python main.py --cnn

# Use DNN model
uv run python main.py --dnn
```

### Controls During Detection

- **Q** - Quit the application
- **S** - Save screenshot of current frame
- **F** - Toggle fullscreen mode

### Command Line Arguments

- `--cnn`: Use CNN model for emotion detection
- `--dnn`: Use DNN model for emotion detection

## 🏗️ Project Structure

```
computer-vision-assignment/
│
├── 📊 Models
│   ├── emotion_cnn_model.h5         # Pre-trained CNN model
│   └── emotion_dnn_model.h5         # Pre-trained DNN model
│
├── 🖥️ Application
│   └── main.py                      # Main application with CLI interface
│
├── 🔧 Training
│   └── train.ipynb                  # Model training notebook
│
├── 🎬 Demo Videos
│   ├── cnn.mp4                      # CNN model demonstration
│   └── dnn.mp4                      # DNN model demonstration
│
└── � Documentation
    └── README.md                    # This file
```

## 🎨 Interface Features

### Real-time Display
- **Enhanced OpenCV interface** with dark overlays
- **Real-time video feed** with emotion detection overlays
- **Model information display** showing current model type
- **Live statistics** including FPS and face count
- **Emotion color coding** for easy identification

### Interactive Controls
- **Fullscreen mode** for immersive experience
- **Screenshot capture** to save detection results
- **Real-time switching** between fullscreen and windowed mode

## 🧠 Model Information

### CNN Model
- **Architecture**: Convolutional layers with pooling and dropout
- **Input**: 128x128 RGB images
- **Output**: 2 emotion classes (Happy/Sad)
- **Performance**: Higher accuracy, more computationally intensive

### DNN Model
- **Architecture**: Fully connected layers with dropout
- **Input**: 128x128 RGB images (flattened)
- **Output**: 2 emotion classes (Happy/Sad)
- **Performance**: Faster inference, suitable for real-time applications

## 📊 Supported Emotions

The system can detect the following 2 emotional states:

1. 😊 **Happy** (Green) - Positive emotions and expressions
2. 😢 **Sad** (Red) - Negative emotions and expressions

## 🔧 Training Custom Models

To train your own models:

1. **Use the training notebook:**
   ```bash
   jupyter notebook train.ipynb
   ```

2. **Follow the notebook instructions** to:
   - Prepare your dataset
   - Configure model parameters
   - Train both CNN and DNN models

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
- Optimized face detection and emotion prediction pipeline
- Automatic frame rate monitoring for performance analysis

## 🔍 Troubleshooting

### Common Issues

1. **Camera not detected:**
   - Check camera permissions
   - Ensure no other applications are using the camera
   - Camera index 0 is used by default

2. **Model loading errors:**
   - Verify model files (`emotion_cnn_model.h5` and `emotion_dnn_model.h5`) exist in the project directory
   - Check TensorFlow/Keras compatibility
   - Try reinstalling dependencies: `uv add tensorflow opencv-python numpy`

3. **Low performance:**
   - Use DNN model instead of CNN for faster processing
   - Close other resource-intensive applications
   - Ensure adequate lighting for better face detection

4. **Display issues:**
   - Press 'F' to toggle fullscreen mode
   - Press 'Q' to quit if the application becomes unresponsive
   - Check display resolution compatibility

### Error Messages

- **"Model file not found"**: Download or train the required model files using `train.ipynb`
- **"Cannot open camera"**: Check camera connection and permissions
- **"ImportError"**: Install missing dependencies with `uv add tensorflow opencv-python numpy`

## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **TensorFlow/Keras** for deep learning framework
- **Face detection datasets** for training data
- **Emotion recognition research** community