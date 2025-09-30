import argparse
import cv2
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import load_model
import time

class LiveEmotionDetector:
    def __init__(self, model_path=None, model_type='cnn'):
        if model_path is None:
            model_path = "/home/sahan/Desktop/computer-vision/facial_recognition_model.keras"
        self.model = load_model(model_path)
        self.model_type = model_type
        
        self.emotion_labels = {
            0: 'Happy',
            1: 'Sad'
        }
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.colors = {
            'Happy': (0, 255, 0),
            'Sad': (255, 0, 0),
            'Angry': (0, 0, 255),
            'Fear': (128, 0, 128),
            'Surprise': (0, 255, 255),
            'Disgust': (0, 128, 0),
            'Neutral': (128, 128, 128)
        }
    
    def preprocess_face(self, face_img):
        face_img = cv2.resize(face_img, (128, 128))
        
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif len(face_img.shape) == 3 and face_img.shape[2] == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        face_img = face_img.astype('float32') / 255.0
        face_img = face_img.reshape(1, 128, 128, 3)
        
        return face_img
    
    def predict_emotion(self, face_img):
        try:
            prediction = self.model.predict(face_img, verbose=0)
            
            happy_confidence = float(prediction[0][0])
            
            if happy_confidence > 0.5:
                emotion_label = 'Happy'
                confidence = happy_confidence
            else:
                emotion_label = 'Sad'
                confidence = 1.0 - happy_confidence
            
            return emotion_label, confidence
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return 'Sad', 0.0
    
    def draw_emotion_info(self, frame, x, y, w, h, emotion, confidence):
        color = self.colors.get(emotion, (255, 255, 255))
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        emotion_text = f"{emotion} ({confidence:.1%})"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(emotion_text, font, font_scale, thickness)
        
        text_x = x
        text_y = y - 10
        
        if text_y < text_h:
            text_y = y + h + text_h + 10
        
        cv2.rectangle(frame, (text_x - 5, text_y - text_h - 5), 
                     (text_x + text_w + 5, text_y + baseline + 5), (0, 0, 0), -1)
        
        cv2.putText(frame, emotion_text, (text_x, text_y), font, font_scale, color, thickness)
    
    def run_detection(self, camera_index=0, display_fps=True):
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Starting emotion detection...")
        print("Press 'q' to quit, 's' to save screenshot, 'f' to toggle fullscreen")
        
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        fullscreen = True
        window_name = 'Live Emotion Detection'
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                frame = cv2.flip(frame, 1)
                
                height, width = frame.shape[:2]
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
                cv2.rectangle(overlay, (0, height-80), (width, height), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                detected_emotions = []
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    
                    processed_face = self.preprocess_face(face_roi)
                    
                    emotion, confidence = self.predict_emotion(processed_face)
                    detected_emotions.append((emotion, confidence))
                    
                    self.draw_emotion_info(frame, x, y, w, h, emotion, confidence)
                
                title_text = "Live Emotion Detection - AI Powered"
                model_text = f"Model: {self.model_type.upper()}"
                
                cv2.putText(frame, title_text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(frame, model_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                if display_fps:
                    fps_counter += 1
                    if fps_counter >= 30:
                        fps_end_time = time.time()
                        current_fps = fps_counter / (fps_end_time - fps_start_time)
                        fps_start_time = fps_end_time
                        fps_counter = 0
                
                fps_text = f"FPS: {current_fps:.1f}"
                faces_text = f"Faces: {len(faces)}"
                
                fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.putText(frame, fps_text, (width - fps_size[0] - 20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, faces_text, (width - fps_size[0] - 20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                controls_text = "Controls: 'Q' - Quit | 'S' - Screenshot | 'F' - Toggle Fullscreen"
                cv2.putText(frame, controls_text, (20, height - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if detected_emotions:
                    primary_emotion = max(detected_emotions, key=lambda x: x[1])
                    status_text = f"Primary Emotion: {primary_emotion[0]} ({primary_emotion[1]:.1%})"
                    status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.putText(frame, status_text, (width - status_size[0] - 20, height - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors.get(primary_emotion[0], (255, 255, 255)), 2)
                else:
                    status_text = "No faces detected"
                    status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.putText(frame, status_text, (width - status_size[0] - 20, height - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"emotion_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved as {filename}")
                elif key == ord('f'):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, 1280, 720)
        
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Emotion Detection")
    parser.add_argument("--dnn", action="store_true", help="Use DNN model")
    parser.add_argument("--cnn", action="store_true", help="Use CNN model")

    args = parser.parse_args()

    if args.dnn:
        print("Using DNN model")
        model_path = "emotion_dnn_model.h5"
        detector = LiveEmotionDetector(model_path, model_type='dnn')
    elif args.cnn:
        print("Using CNN model")
        model_path = "emotion_cnn_model.h5"
        detector = LiveEmotionDetector(model_path, model_type='cnn')
    else:
        print("No model specified. Using CNN model by default...")
        model_path = "emotion_cnn_model.h5"
        detector = LiveEmotionDetector(model_path, model_type='cnn')

    detector.run_detection(camera_index=0, display_fps=True)

