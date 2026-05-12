import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Using a small model for the demo
video_path = 'demo_lockon_widefix.mp4'
cap = cv2.VideoCapture(video_path)

frame_idx = 0
targets = [0, 30, 60, 90]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(frame, persist=True, tracker='bytetrack.yaml', conf=0.18, classes=[0], imgsz=640, verbose=False)
    
    if frame_idx in targets:
        result = results[0]
        boxes = result.boxes
        num_boxes = len(boxes)
        has_ids = boxes.id is not None
        id_count = len(set(boxes.id.int().tolist())) if has_ids else 0
        print(f"Frame {frame_idx}: Boxes={num_boxes}, Has IDs={has_ids}, ID Count={id_count}")
        
    frame_idx += 1
    if frame_idx > 90:
        break

cap.release()
