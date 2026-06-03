"""
3. Predict Module
Load a pre-trained model and predict a single image.
"""

import os
import cv2
import torch
import numpy as np
import argparse
import joblib

from config import RESULTS_DIR, CLASS_NAMES, IMG_SIZE
from model import preprocess_input
from preprocessing import background_cancellation
import preprocessing
from statistical_features import extract_statistical_features

def predict_ml(image_path, model_path):
    print(f"Loading ML model from {model_path}...")
    pipeline = joblib.load(model_path)
    model = pipeline['model']
    scaler = pipeline['scaler']
    le = pipeline['le']
    color_space = pipeline['color_space']
    
    img = cv2.imread(image_path)
    if img is None:
        print("Cannot read image.")
        return
        
    print("Preprocessing image (background cancellation)...")
    roi = background_cancellation(img)
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    img_cs = preprocessing.convert_color_spaces(roi_rgb)[color_space]
    features = extract_statistical_features(img_cs)
    
    # Scale
    features_sc = scaler.transform([features])
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_sc)[0]
    else:
        probs = None
        
    pred_idx = model.predict(features_sc)[0]
    predicted_class = le.inverse_transform([pred_idx])[0]
    
    print("\n" + "="*40)
    print("      PREDICTION RESULTS (ML)")
    print("="*40)
    print(f"Input: {image_path}")
    print(f"Model: {type(model).__name__} (Color Space: {color_space})")
    print(f"Predicted Class: {predicted_class}")
    if probs is not None:
        print("Probabilities:")
        for i, cls in enumerate(le.classes_):
            print(f" - {cls}: {probs[i]*100:.2f}%")
    print("="*40)

def predict_cnn(image_path, model_path, model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(CLASS_NAMES)
    
    color_space = 'RGB'
    if model_type == "custom_cnn":
        from model import CustomCNN
        model = CustomCNN(num_classes).to(device)
    elif model_type == "mobilenet":
        from model import MobileNetV3Edge
        model = MobileNetV3Edge(num_classes).to(device)
    elif model_type == "papercnn":
        from model import PaperCNN
        model = PaperCNN(num_classes).to(device)
        # Infer color space from path
        import re
        match = re.search(r'results_([a-zA-Z]+)/', model_path)
        if match:
            color_space = match.group(1)
            print(f"Inferred color space: {color_space}")
            
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    img = cv2.imread(image_path)
    if img is None:
        print("Cannot read image.")
        return
    
    print("Preprocessing image (background cancellation)...")
    roi = background_cancellation(img)
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    if model_type == "papercnn" and color_space != 'RGB':
        roi_rgb = preprocessing.convert_color_spaces(roi_rgb)[color_space]
        
    # Expand dims from (H, W, 3) to (1, H, W, 3) for standardized preprocess
    batch_img = np.expand_dims(roi_rgb, axis=0) 
    batch_img_p = preprocess_input(batch_img)
    tensor_img = torch.tensor(batch_img_p).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(tensor_img)
        # Apply Softmax for probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        
    predicted_idx = np.argmax(probs)
    predicted_class = CLASS_NAMES[predicted_idx]
    
    print("\n" + "="*40)
    print("      PREDICTION RESULTS (CNN)")
    print("="*40)
    print(f"Input: {image_path}")
    print(f"Model Type: {model_type}")
    print(f"Predicted Class: {predicted_class}")
    print("Probabilities:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f" - {cls}: {probs[i]*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Tomato Quality using the BEST model")
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
        exit(1)
        
    info_path = os.path.join(RESULTS_DIR, "best_model_info.json")
    if not os.path.exists(info_path):
        print(f"Error: Could not find {info_path}. Please run train_cachua_module.py first.")
        exit(1)
        
    import json
    with open(info_path, 'r') as f:
        best_info = json.load(f)
        
    model_path = best_info['model_path']
    model_type = best_info['model_type']
            
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        exit(1)
        
    print("\n" + "="*60)
    print("  LOADING THE ABSOLUTE BEST MODEL")
    print(f"  Method: {best_info['model_name']} ({best_info['color_space']})")
    print(f"  Accuracy: {best_info['accuracy']:.2f}%")
    print("="*60 + "\n")
        
    if model_type == "ml":
        predict_ml(args.image, model_path)
    else:
        predict_cnn(args.image, model_path, model_type)
