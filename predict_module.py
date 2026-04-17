"""
3. Predict Module
Load a pre-trained model and predict a single image.
"""

import os
import cv2
import torch
import numpy as np
import argparse

from config import RESULTS_DIR, CLASS_NAMES, IMG_SIZE
from model import CustomCNN, preprocess_input
from preprocessing import background_cancellation

def predict_single_image(image_path, model_path):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    num_classes = len(CLASS_NAMES)
    model = CustomCNN(num_classes).to(device)
    
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # the checkpoint has 'model_state_dict', but check if it's directly saved
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Preprocess Image
    img = cv2.imread(image_path)
    if img is None:
        print("Cannot read image.")
        return
    
    print("Preprocessing image (background cancellation)...")
    roi = background_cancellation(img)
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
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
    print("      PREDICTION RESULTS")
    print("="*40)
    print(f"Input: {image_path}")
    print(f"Predicted Class: {predicted_class}")
    print("Probabilities:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f" - {cls}: {probs[i]*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Tomato Quality")
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default=None, help="Path to .pth checkpoint")
    args = parser.parse_args()
    
    # default model to transfer_save_model
    if args.model is None:
        args.model = os.path.join(RESULTS_DIR, "transfer_save_model", "transfer_cnn_best.pth")
        
    predict_single_image(args.image, args.model)
