from ultralytics import YOLO
import cv2
# 1. Gọi "bộ não" tốt nhất của bạn lên
model = YOLO('weights/best.pt')

# 2. Chọn nguồn hình ảnh (Mở Webcam của laptop thì để số 0)
# NẾU bạn có 1 file video, hãy đổi số 0 thành tên file, ví dụ: 'video_truong_hoc.mp4'
nguon_hinh_anh = r"C:\Users\Lenovo\Desktop\codeNCKH\codeDot1\violenceNewV26_rl coppyy\RWF-2000\train\Fight\QUGT4a6qcJs_2.avi"

# 3. Bắt đầu quét và hiển thị lên màn hình
# conf=0.5 nghĩa là: AI chắc chắn trên 50% mới được vẽ khung
model.predict(source=nguon_hinh_anh, save=True, conf=0.3, iou=0.6)

# Lưu ý: Khi khung hình hiện lên, để tắt nó đi, bạn bấm phím 'q' trên bàn phím nhé!