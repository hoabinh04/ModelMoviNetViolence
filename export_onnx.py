import torch
import cv2
import os
import sys

# Import UltraLytics for YOLO export
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not installed. Please install it with 'pip install ultralytics'")
    sys.exit(1)

# Modify import path to access src folder
# Add both potential src locations to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ViolenceDetection_Final'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ViolenceDetection_Final', 'src'))

try:
    from src.buoc3_model import MobileNetV3TSMLite
except ImportError:
    try:
        from buoc3_model import MobileNetV3TSMLite
    except ImportError:
        print("Could not import MobileNetV3TSMLite from src.buoc3_model")
        sys.exit(1)


def export_yolo():
    print("Exporting YOLOv8 model to ONNX...")
    # Adjust path if needed based on the file locations
    yolo_paths = [
        "src/yolo26n.pt",
        "ViolenceDetection_Final/src/yolo26n.pt"
    ]
    
    yolo_path = None
    for p in yolo_paths:
        if os.path.exists(p):
            yolo_path = p
            break
            
    if yolo_path is None:
        print("Could not find yolo26n.pt")
        return
        
    model = YOLO(yolo_path)
    # Target img_size for fast processing on edge devices
    # imgsz=320 is typical for RPi 
    model.export(format="onnx", imgsz=320, simplify=True)
    print(f"YOLO exported successfully to ONNX at {os.path.splitext(yolo_path)[0]}.onnx")


def export_tsm():
    print("Exporting TSM model to ONNX...")
    
    # Need to match the config of the trained model
    num_segments = 12 # Common in this project
    img_size = 224 # Standard crop size
    num_classes = 2 # Fight / NonFight
    
    tsm_paths = [
        "weights/best_tsm_topdown.pth",
        "ViolenceDetection_Final/weights/best_tsm_topdown.pth"
    ]
    
    tsm_path = None
    for p in tsm_paths:
        if os.path.exists(p):
            tsm_path = p
            break
            
    if tsm_path is None:
        print("Could not find best_tsm_topdown.pth")
        return

    # Load model
    model = MobileNetV3TSMLite(num_classes=num_classes, num_segments=num_segments, pretrained=False)
    
    # Load weights
    try:
        checkpoint = torch.load(tsm_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
        
    model.eval()

    # Create dummy input for tracing
    # (batch_size * num_segments, channels, height, width)
    # We export for batch_size=1
    dummy_input = torch.randn(1 * num_segments, 3, img_size, img_size)

    output_path = os.path.splitext(tsm_path)[0] + ".onnx"

    print(f"Exporting TSM to {output_path} (this may take a moment)...")

    # Export
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True,
        opset_version=11, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print("TSM exported successfully!")


if __name__ == "__main__":
    export_yolo()
    export_tsm()
    print("Done! You can now copy the .onnx files to your Raspberry Pi.")
