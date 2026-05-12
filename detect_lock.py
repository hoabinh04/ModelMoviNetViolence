import cv2
import os
import numpy as np

video_path = "demo_lockon_widefix.mp4"
output_dir = "tmp_lock_active_final"
cap = cv2.VideoCapture(video_path)

active_lock_frames = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # ROI x:[0,900], y:[0,45]
    roi = frame[0:45, 0:900]
    
    # BGR threshold (B<80, G in [120,230], R>200)
    # OpenCV uses BGR
    mask = (roi[:, :, 0] < 80) & (roi[:, :, 1] >= 120) & (roi[:, :, 1] <= 230) & (roi[:, :, 2] > 200)
    
    # Consider "significant orange pixels" as > 10 pixels to avoid noise, 
    # but the prompt says detect frames. Let's use a small threshold.
    if np.sum(mask) > 20: 
        active_lock_frames.append(frame_idx)
    
    frame_idx += 1

cap.release()

print(f"Total active-lock frames: {len(active_lock_frames)}")
print(f"Active-lock frame indices: {active_lock_frames}")

if active_lock_frames:
    # Save 8 representative frames
    step = max(1, len(active_lock_frames) // 8)
    for i in range(8):
        idx = active_lock_frames[min(i * step, len(active_lock_frames) - 1)]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f"lock_{idx}.jpg"), frame)
        cap.release()
