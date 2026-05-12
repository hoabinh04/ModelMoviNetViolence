import cv2
import os

video_path = r"C:\Users\Lenovo\Desktop\NewVioDe\demo_lockon_widefix.mp4"
output_dir = r"C:\Users\Lenovo\.gemini\antigravity\brain\ba37f367-9edb-4135-9329-6c370b54077e\artifacts\frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 30 == 0:
        out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path, frame)
        saved_count += 1
    frame_idx += 1

cap.release()
print(f"Saved {saved_count} frames to {output_dir}")
