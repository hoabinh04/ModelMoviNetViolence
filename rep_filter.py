import cv2
from ultralytics import YOLO
from collections import Counter

# Constants from buoc6
HUMAN_MIN_HEIGHT_RATIO = 0.045
HUMAN_MIN_ASPECT = 0.22
HUMAN_MAX_ASPECT = 4.8
HUMAN_MIN_AREA_RATIO = 0.0007
LOW_CONF_BOX_THRESHOLD = 0.22
LOW_CONF_MIN_HEIGHT_RATIO = 0.055
ALT_CLASS_MIN_CONF = 0.20
ALT_CLASS_MIN_HEIGHT_RATIO = 0.055
ALT_CLASS_MIN_ASPECT = 0.22
WIDE_BOX_MIN_HEIGHT_RATIO = 0.07
WIDE_BOX_MIN_AREA_RATIO = 0.012

# HARDCODED: If HUMAN_MAX_AREA_RATIO not found, use a large default or re-check
HUMAN_MAX_AREA_RATIO = 1.0 

def replicate_filter(box, cls, conf, frame_w, frame_h):
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    aspect = box_h / box_w if box_w > 0 else 0
    area_ratio = (box_w * box_h) / (frame_w * frame_h)
    h_ratio = box_h / frame_h

    # Logic based on buoc6_main_pipeline summary
    # 1. Height filter
    if h_ratio < HUMAN_MIN_HEIGHT_RATIO:
        return False, "h_min"
    
    # 2. Aspect filter
    if aspect < HUMAN_MIN_ASPECT or aspect > HUMAN_MAX_ASPECT:
        return False, "aspect"
    
    # 3. Area filter
    if area_ratio < HUMAN_MIN_AREA_RATIO or area_ratio > HUMAN_MAX_AREA_RATIO:
        return False, "area"

    # 4. Low conf rules (class 0)
    if cls == 0:
        if conf < LOW_CONF_BOX_THRESHOLD and h_ratio < LOW_CONF_MIN_HEIGHT_RATIO:
            return False, "low_conf_h"
    
    # 5. Alt class rules (18, 72)
    if cls in [18, 72]:
        if conf < ALT_CLASS_MIN_CONF:
            return False, "alt_conf"
        if h_ratio < ALT_CLASS_MIN_HEIGHT_RATIO:
            return False, "alt_h"
        if aspect < ALT_CLASS_MIN_ASPECT:
            return False, "alt_aspect"
            
    # 6. Hardcoded extra from search
    if h_ratio < 0.065 or area_ratio < 0.008:
         # Note: The buoc6 script seemed to have some specific 'and' conditions for wide boxes
         # but let's stick to the basics of the primary pipeline filters found.
         pass

    return True, "keep"

model = YOLO('src/yolo26n.pt')
video_path = 'C:/Users/Lenovo/Desktop/codeNCKH/codeDot2/Data_Violence_Detection/Violence/f1.mp4'
cap = cv2.VideoCapture(video_path)

total_rejections = Counter()
frame_idx = 0

print(f"{'Frame':<6} | {'Raw':<4} | {'Kept':<4} | {'Details'}")
print("-" * 50)

while cap.isOpened() and frame_idx < 40:
    ret, frame = cap.read()
    if not ret:
        break
    
    h_frame, w_frame = frame.shape[:2]
    results = model.track(frame, persist=True, tracker='bytetrack.yaml', conf=0.18, classes=[0,18,72], imgsz=640, verbose=False)
    
    boxes = results[0].boxes
    raw_count = len(boxes)
    kept_count = 0
    
    for i in range(raw_count):
        box = boxes.xyxy[i].tolist()
        cls = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        
        passed, reason = replicate_filter(box, cls, conf, w_frame, h_frame)
        if passed:
            kept_count += 1
        else:
            total_rejections[reason] += 1
            
    print(f"{frame_idx:<6} | {raw_count:<4} | {kept_count:<4}")
    frame_idx += 1

cap.release()
print("\nAggregate Rejection Reasons:")
for reason, count in total_rejections.items():
    print(f"{reason}: {count}")

