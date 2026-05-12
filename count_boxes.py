from ultralytics import YOLO
import cv2

model = YOLO('src/yolo26n.pt')
video_path = 'C:/Users/Lenovo/Desktop/codeNCKH/codeDot2/Data_Violence_Detection/Violence/f88.mp4'
conf = 0.18
classes = [0]
max_frames = 340

results = model.track(source=video_path, stream=True, persist=True, tracker='bytetrack.yaml', conf=conf, classes=classes)

histogram = {0: 0, 1: 0, '2+': 0}
frames_ge_2 = []

for i, r in enumerate(results):
    if i >= max_frames:
        break
    
    count = len(r.boxes)
    if count == 0:
        histogram[0] += 1
    elif count == 1:
        histogram[1] += 1
    else:
        histogram['2+'] += 1
        if len(frames_ge_2) < 20:
            frames_ge_2.append(i)

print(f"Histogram: {histogram}")
print(f"First 20 frames with boxes>=2: {frames_ge_2}")
