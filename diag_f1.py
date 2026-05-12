import cv2
import json
import os
import torch
from ultralytics import YOLO
import sys

# Constants from src/buoc6_main_pipeline.py
HUMAN_MIN_HEIGHT_RATIO = 0.045
HUMAN_MIN_ASPECT = 0.22
HUMAN_MAX_ASPECT = 1.2 # Inferred from context or usual values
HUMAN_MIN_AREA_RATIO = 0.0007
HUMAN_MAX_AREA_RATIO = 0.15 # Inferred
LOW_CONF_BOX_THRESHOLD = 0.22
LOW_CONF_MIN_HEIGHT_RATIO = 0.055
ALT_CLASS_MIN_CONF = 0.20
ALT_CLASS_MIN_HEIGHT_RATIO = 0.055
ALT_CLASS_MIN_ASPECT = 0.22

def analyze_video(video_path, output_path):
    # Detectors used in the pipeline
    # Note: buoc6_main_pipeline uses YOLO("yolo11n.pt") usually
    try:
        model = YOLO("yolo11n.pt")
    except:
        model = YOLO("yolov8n.pt")

    stats = {
        "total_frames": 0,
        "raw_boxes_total": 0,
        "kept_boxes_total": 0,
        "kept_frames": 0,
        "reject_counts": {
            "h_min": 0, "area_low": 0, "area_high": 0, "aspect": 0,
            "low_conf_h": 0, "alt_conf": 0, "alt_h": 0, "alt_aspect": 0
        },
        "per_frame_kept_first40": []
    }
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return

    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_area = w_frame * h_frame

    frame_idx = 0
    while frame_idx < 120:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Results from detection
        results = model.track(frame, persist=True, conf=0.18, show=False, verbose=False)
        
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes else []
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes else []
        ids = results[0].boxes.id.cpu().numpy().astype(int) if (results[0].boxes is not None and results[0].boxes.id is not None) else [0]*len(boxes)

        stats["total_frames"] += 1
        stats["raw_boxes_total"] += len(boxes)
        
        kept_this_frame = 0
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = box
            box_h = y2 - y1
            box_w = x2 - x1
            area_ratio = (box_w * box_h) / total_area
            aspect = box_w / box_h if box_h > 0 else 0
            
            # Filtering Logic from src/buoc6_main_pipeline.py
            if box_h < h_frame * HUMAN_MIN_HEIGHT_RATIO:
                stats["reject_counts"]["h_min"] += 1
                continue
            if area_ratio < HUMAN_MIN_AREA_RATIO:
                stats["reject_counts"]["area_low"] += 1
                continue
            if area_ratio > HUMAN_MAX_AREA_RATIO:
                stats["reject_counts"]["area_high"] += 1
                continue
            if aspect < HUMAN_MIN_ASPECT or aspect > 1.4: # Using 1.4 as a safe max aspect if not found
                stats["reject_counts"]["aspect"] += 1
                continue
            
            if float(conf) < LOW_CONF_BOX_THRESHOLD and box_h < h_frame * LOW_CONF_MIN_HEIGHT_RATIO:
                stats["reject_counts"]["low_conf_h"] += 1
                continue
            
            if cls_id != 0:
                if float(conf) < ALT_CLASS_MIN_CONF:
                    stats["reject_counts"]["alt_conf"] += 1
                    continue
                if box_h < h_frame * ALT_CLASS_MIN_HEIGHT_RATIO:
                    stats["reject_counts"]["alt_h"] += 1
                    continue
                if aspect < ALT_CLASS_MIN_ASPECT:
                    stats["reject_counts"]["alt_aspect"] += 1
                    continue
            
            kept_this_frame += 1
        
        stats["kept_boxes_total"] += kept_this_frame
        if kept_this_frame > 0:
            stats["kept_frames"] += 1
        
        if frame_idx < 40:
            stats["per_frame_kept_first40"].append(kept_this_frame)
            
        frame_idx += 1

    cap.release()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(output_path)

if __name__ == "__main__":
    analyze_video("Violence/f1.mp4", "tmp_batch_stats/filter_diag_f1.json")
