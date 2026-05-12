from ultralytics import YOLO
import cv2
import numpy as np
import json

videos = [
    r'C:/Users/Lenovo/Desktop/codeNCKH/codeDot2/Data_Violence_Detection/Violence/f1.mp4',
    r'C:/Users/Lenovo/Desktop/codeNCKH/codeDot2/Data_Violence_Detection/Violence/f19.mp4',
    r'C:/Users/Lenovo/Desktop/codeNCKH/codeDot2/Data_Violence_Detection/Violence/f99.mp4',
    r'C:/Users/Lenovo/Desktop/codeNCKH/codeDot2/Data_Violence_Detection/Violence/f279.mp4',
    r'C:/Users/Lenovo/Desktop/codeNCKH/codeDot2/Data_Violence_Detection/NonViolence/nf343.mp4'
]

model = YOLO('src/yolo26n.pt')
results = []

for video_path in videos:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Cannot open {video_path}')
        continue
    
    total_frames = 0
    frames_with_boxes = 0
    total_boxes = 0
    frames_with_2plus_boxes = 0
    class_counts = {0: 0, 18: 0, 72: 0}
    
    while total_frames < 300:
        ret, frame = cap.read()
        if not ret:
            break
            
        total_frames += 1
        prediction = model.predict(frame, conf=0.18, classes=[0, 18, 72], verbose=False, imgsz=640)[0]
        
        boxes = prediction.boxes
        num_boxes = len(boxes)
        
        if num_boxes > 0:
            frames_with_boxes += 1
            total_boxes += num_boxes
            if num_boxes >= 2:
                frames_with_2plus_boxes += 1
            
            for cls_tensor in boxes.cls:
                cls = int(cls_tensor.item())
                if cls in class_counts:
                    class_counts[cls] += 1
                    
    cap.release()
    
    results.append({
        'video': video_path,
        'total_frames': total_frames,
        'frames_with_boxes': frames_with_boxes,
        'mean_boxes_per_frame': total_boxes / total_frames if total_frames > 0 else 0,
        'frames_with_2plus_boxes': frames_with_2plus_boxes,
        'class_counts': class_counts
    })

print(json.dumps(results, indent=2))
