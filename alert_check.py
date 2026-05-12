import cv2
import numpy as np
import os

video_path = 'demo_lockon_widefix.mp4'
output_dir = 'tmp_alert_after_override'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)
alert_frames = []

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Crop ROI x:[0,340], y:[630,719]
    # Note: numpy slicing is [y1:y2, x1:x2]
    roi = frame[630:720, 0:341]
    
    # Red pixels (R>150, G<120, B<120)
    # OpenCV uses BGR
    red_mask = (roi[:,:,2] > 150) & (roi[:,:,1] < 120) & (roi[:,:,0] < 120)
    red_count = np.sum(red_mask)
    
    # Green pixels (G>150, R<140, B<140)
    green_mask = (roi[:,:,1] > 150) & (roi[:,:,2] < 140) & (roi[:,:,0] < 140)
    green_count = np.sum(green_mask)
    
    # ALERT condition: red_count > 130 and red_count > green_count * 0.85
    if red_count > 130 and red_count > (green_count * 0.85):
        alert_frames.append(frame_idx)
    
    frame_idx += 1

cap.release()

print(f"alert_frame_count: {len(alert_frames)}")
if alert_frames:
    print(f"First 15: {alert_frames[:15]}")
    print(f"Last 15: {alert_frames[-15:]}")
    
    # Save 6 representative alert frames
    indices_to_save = np.linspace(0, len(alert_frames) - 1, 6, dtype=int)
    # Reload video to grab specific frames efficiently or just use the list
    cap = cv2.VideoCapture(video_path)
    save_idx = 0
    for idx_in_alert in indices_to_save:
        actual_frame_idx = alert_frames[idx_in_alert]
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f"alert_{actual_frame_idx}.jpg"), frame)
    cap.release()
