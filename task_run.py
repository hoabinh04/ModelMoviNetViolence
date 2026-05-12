import cv2
from ultralytics import YOLO

def process_video(video_path):
    model = YOLO('yolov8n.pt')  # Assuming yolov8n as default if not specified
    results = model.track(source=video_path, conf=0.18, persist=True, tracker='bytetrack.yaml', classes=[0], imgsz=640, stream=True)
    
    for i, res in enumerate(results):
        if i in [0, 30, 60, 90]:
            # Exact extraction logic requested
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes else []
            ids = res.boxes.id.cpu().numpy() if (res.boxes and res.boxes.id is not None) else []
            
            print(f"Frame {i}: len(boxes)={len(boxes)}, len(ids)={len(ids)}")
        if i > 90:
            break

if __name__ == "__main__":
    video_path = "f88.mp4"
    import os
    if os.path.exists(video_path):
        process_video(video_path)
    else:
        print(f"File {video_path} not found in {os.getcwd()}")
