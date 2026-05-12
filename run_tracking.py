import cv2
from ultralytics import YOLO

def main():
    video_path = r'C:/Users/Lenovo/Desktop/codeNCKH/codeDot2/Data_Violence_Detection/Violence/f88.mp4'
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)
    
    target_frames = [0, 30, 60, 90, 120, 150]
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count in target_frames:
            results = model.track(frame, persist=True, tracker='bytetrack.yaml', conf=0.18, classes=[0], imgsz=640, verbose=False)
            
            boxes = results[0].boxes
            ids = boxes.id.tolist() if boxes.id is not None else []
            print(f'Frame {frame_count:3d}: Boxes={len(boxes)}, IDs={len(ids)}')
        
        frame_count += 1
        if frame_count > 150:
            break
            
    cap.release()

if __name__ == '__main__':
    main()
