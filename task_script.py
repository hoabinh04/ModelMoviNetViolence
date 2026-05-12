import cv2
from ultralytics import YOLO

model = YOLO('src/yolo26n.pt')
video_path = r'C:\Users\Lenovo\Desktop\codeNCKH\codeDot2\Data_Violence_Detection\Violence\f88.mp4'
cap = cv2.VideoCapture(video_path)

count_a = 0
count_b = 0
count_c = 0
indices_a = []

frame_idx = 0
while frame_idx < 200:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(frame, persist=True, tracker='bytetrack.yaml', conf=0.18, classes=[0], imgsz=640, verbose=False)
    boxes = results[0].boxes
    ids = boxes.id

    if boxes is not None and len(boxes) > 0:
        if ids is None:
            count_a += 1
            if len(indices_a) < 10:
                indices_a.append(frame_idx)
        else:
            count_b += 1
    else:
        count_c += 1
    
    frame_idx += 1

print(f'Case (a) count: {count_a}')
print(f'Case (b) count: {count_b}')
print(f'Case (c) count: {count_c}')
print(f'First 10 indices for case (a): {indices_a}')

cap.release()
