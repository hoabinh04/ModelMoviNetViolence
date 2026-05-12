# Chi tiết Mã Nguồn (Source Code) - Antigravity Violence Detection

Thư mục `src/` chứa toàn bộ mã nguồn cốt lõi của hệ thống phát hiện bạo lực, bao gồm cả giai đoạn huấn luyện (training) và suy luận thực tế (inference). Dưới đây là tài liệu kỹ thuật chi tiết giải thích vai trò của từng tệp, các hàm quan trọng và logic cốt lõi.

## 1. Cấu trúc Dự án (Project Directory Structure)
Dự án được tổ chức theo chuẩn modular để AI dễ dàng định vị, đọc hiểu và bảo trì:
*   `ViolenceDetection_Final/`: Thư mục gốc chứa toàn bộ dự án bàn giao.
    *   `src/`: Thư mục chứa mã nguồn cốt lõi (Model, Pipeline, Kinematics, Dataloader).
    *   `weights/`: Nơi lưu trữ bộ não của hệ thống (Trọng số TSM `.pth`, cấu hình `.json`).
    *   `winform_app.py`: Tệp khởi chạy giao diện Desktop (Sử dụng PyQt6).
    *   `ui_app.py`: Tệp khởi chạy giao diện Web (Sử dụng Gradio).
    *   `requirements.txt`: Danh sách chính xác các thư viện cần cài đặt.

## 2. Các Thư viện Cốt lõi (Tech Stack & Dependencies)
Hệ thống vận hành trơn tru dựa trên sự kết hợp của các thư viện (đã kiểm chứng ổn định):
*   **PyTorch (`torch`, `torchvision`)**: Nền tảng Deep Learning dùng để xây dựng và suy luận mô hình TSM. Ưu tiên chạy trên GPU (CUDA) thông qua AMP (FP16).
*   **Ultralytics (`ultralytics`)**: Framework mạnh mẽ chứa mô hình YOLO (tùy chỉnh lại từ YOLOv8 Nano) dùng cho việc phát hiện và Tracking (ByteTrack) người.
*   **OpenCV (`cv2`)**: Thư viện thị giác máy tính dùng để đọc video, vẽ Bounding Box, làm mượt hình ảnh, thay đổi kích thước (Letterbox).
*   **Gradio (`gradio`) & PyQt6 (`PyQt6`)**: Xây dựng giao diện người dùng (UI) tương ứng cho nền tảng Web và Windows Desktop.
*   **NumPy (`numpy`)**: Xử lý ma trận dữ liệu và tính toán vật lý (trong Kinematics).

## 3. Quy tắc Viết Code (AI Handover Guidelines)
**CẢNH BÁO QUAN TRỌNG CHO AI**: Tài liệu này là kim chỉ nam cho các phiên làm việc (sessions) tiếp theo. Khi AI "thức dậy" và đọc dự án này, vui lòng tuân thủ tuyệt đối các quy tắc sau:
1.  **Tách biệt Logic (Separation of Concerns)**: Code UI (`winform_app.py`, `ui_app.py`) tuyệt đối chỉ làm nhiệm vụ lấy tham số từ người dùng. Mọi logic nhận diện phải được truyền vào và xử lý tại `src/buoc6_main_pipeline.py`.
2.  **Ưu tiên Tracking**: Luôn giữ mặc định `detector-mode="track"` (ByteTrack) cho YOLO để đảm bảo tính ổn định của ROI, tuyệt đối không dùng `predict` khiến Bounding Box bị "nhảy loạn" (Jitter).
3.  **Tôn trọng Hysteresis & Smoothing**: Các hằng số như `ROI_SIZE_SMOOTH (0.15)`, `SEARCH_CENTER_ALPHA (0.15)`, và các `STREAK` đã được tinh chỉnh hoàn hảo để chống báo động giả. Không được tự ý thay đổi trừ khi người dùng yêu cầu chỉnh độ nhạy.
4.  **Hiển thị Letterbox**: Mọi luồng video xuất ra màn hình UI phải đi qua hàm `letterbox_resize` để giữ nguyên tỉ lệ người thực tế (Aspect Ratio), không dùng `cv2.resize` thuần túy làm móp méo hình ảnh.
5.  **Quy trình Ghi nhận và Sửa Lỗi (Error Tracking Loop)**: 
    *   Mọi phát hiện về lỗi (False Negative/Positive, Bounding box lỗi, Runtime Error) phải được AI phân tích và ghi chi tiết vào file **`src/errors.md`**.
    *   Mọi phương án khắc phục, thay đổi biến, cấu trúc lệnh (Workaround & Fixes) phải được tài liệu hóa vào **`src/errorsfix.md`**.
    *   **BẮT BUỘC**: Bất kỳ AI nào thức dậy nhận dự án phải kiểm tra 3 file: `README.md`, `errors.md`, và `errorsfix.md` ĐẦU TIÊN để đồng bộ hóa trí nhớ (Context Sync) trước khi đụng vào code, tránh lặp lại lỗi cũ.

## 4. Tình trạng Hiện tại (Current Status)
*   **Pipeline Stable**: Hệ thống đang sử dụng `buoc6_main_pipeline_stable.py` (bản chuẩn) làm bộ não chính, hoạt động cực kỳ mượt mà.
*   **Giao diện**: Sử dụng `winform_app_v2.py` (đã fix lỗi FileNotFoundError bằng Path tuyệt đối).
*   **Lỗi đang theo dõi**: Mô hình TSM có dấu hiệu bỏ lọt hành vi bạo lực (False Negative) trên tập dữ liệu SCVD khi góc quay từ trên xuống khiến YOLO mất dấu đối tượng (`det=00`). Các phương án nới lỏng `det_conf` và ép dùng `Full-frame Rescue` đang được ghi nhận tại `errorsfix.md`.

---

## 5. Luồng Logic Hoạt Động Tổng Thể

Hệ thống hoạt động dựa trên sự kết hợp của 2 mô hình (Hybrid Approach):
1.  **Phát hiện đối tượng (YOLO)**: Tìm vị trí con người trong khung hình và gán ID theo dõi (Tracking).
2.  **Nhận diện hành vi (TSM)**: Phân tích một chuỗi các khung hình (segments) của các đối tượng nghi ngờ (ROI) để đưa ra xác suất bạo lực.
3.  **Hậu xử lý (Post-processing)**: Sử dụng các bộ lọc vật lý (Kinematics), làm mượt khung hình (Smoothing), và thích nghi đám đông (Crowd Adaptation) để ra quyết định cuối cùng.

---

## 5. Chi tiết các Tệp tin (Files Breakdown)

### 🌟 1. `buoc6_main_pipeline.py` (Thành phần Quan trọng nhất - Inference)
Đây là "trái tim" của hệ thống khi chạy thực tế. Tệp này lấy luồng video đầu vào, chạy YOLO để track người, cắt các vùng nghi ngờ (ROI), và đưa vào TSM để dự đoán.

**Các tính năng & Logic nổi bật:**
*   **Tracking & Khóa mục tiêu (Lock-on)**: Sử dụng thuật toán ByteTrack của YOLO để bám theo người. Khi phát hiện các đối tượng có tương tác gần, hệ thống sẽ gộp chúng vào chung một vùng cắt (tube) để quan sát sự kiện.
*   **Làm mượt khung hình (ROI Smoothing)**: Tránh hiện tượng khung hình "nhảy loạn" (jitter) khi đối tượng đứng yên bằng cách dùng Exponential Smoothing (`ROI_SIZE_SMOOTH`, `SEARCH_CENTER_ALPHA`). 
*   **Chống méo hình (Letterbox Resizing)**: Ép kích thước khung hình hiển thị về một chuẩn cố định (VD: 720x720) thông qua hàm `letterbox_resize()`, chèn viền đen nếu cần để không làm méo tỷ lệ cơ thể người.
*   **Hệ thống Cứu nguy (Full-frame Rescue)**: Nếu vùng cắt theo dõi đối tượng quá nhỏ hoặc không rõ ràng, hệ thống tự động kiểm tra toàn bộ khung hình tổng để không bỏ sót hành vi bạo lực ở bối cảnh rộng.
*   **Thích nghi đám đông (Crowd Adaptive Tuning)**: Tự động tính toán mật độ người trong khung hình (`crowd_factor`). Nếu quá đông, ngưỡng báo động sẽ được điều chỉnh tự động để giảm báo động giả.

### 🌟 2. `train_tsm_topdown.py` (Script Huấn luyện)
Đảm nhiệm việc huấn luyện (fine-tune) mô hình nhận diện hành vi (TSM) trên tập dữ liệu video.

**Các tính năng nổi bật:**
*   **Hard-Negative Mining**: Cơ chế tự động học lại những video "bình thường nhưng dễ nhầm thành bạo lực" (hard negatives) bằng cách nhân bản chúng lên (`hard_negative_repeat`) trong quá trình huấn luyện, giúp mô hình bớt nhạy cảm sai.
*   **Lấy mẫu khung hình (Frame Sampling)**: Lớp `TopdownVideoDataset` trích xuất `num_segments` (ví dụ 12 khung hình) rải đều hoặc ngẫu nhiên từ một video để đại diện cho toàn bộ hành động.
*   **Tối ưu tốc độ (AMP)**: Sử dụng Mixed Precision (FP16) thông qua `torch.cuda.amp` để train nhanh hơn và tiết kiệm bộ nhớ VRAM.

### 🧠 3. `buoc3_model.py` (Kiến trúc Mô hình TSM)
Định nghĩa cấu trúc mạng **TSM (Temporal Shift Module)** kết hợp với **MobileNetV3**.

*   **Logic cốt lõi của TSM**: Thay vì dùng mô hình 3D CNN quá nặng, TSM dịch chuyển (shift) một phần dữ liệu (1/8 số kênh) dọc theo trục thời gian giữa các khung hình liên tiếp. Việc này giúp mạng 2D CNN thông thường có thể "nhìn thấy" chiều thời gian và hiểu được hành động (tay đang giơ lên hay hạ xuống) mà chi phí tính toán gần như không đổi.
*   **Hỗ trợ đa cấu trúc**: Lớp `MobileNetV3TSMLite` hỗ trợ chuyển đổi linh hoạt qua tham số `variant` giữa `small` (nhẹ, nhanh) và `large` (chính xác cao - bản đang được sử dụng chính).

### ⚙️ 4. `buoc5_kinematics.py` (Lọc nhiễu Vật lý)
Xử lý các tình huống cảnh báo giả (False Positives) dựa trên các quy luật vật lý cơ bản.

*   **Logic**: Lớp `KinematicsGate` theo dõi tốc độ (`v`) và gia tốc (`a`) của các đối tượng. Nếu TSM báo động là bạo lực, nhưng những người đó di chuyển quá chậm (tốc độ < `GATE_V_THRESHOLD`) hoặc không có chuyển động giật cục, hệ thống sẽ "phủ quyết" cảnh báo đó, giúp phân biệt rõ giữa "ôm nhau/nói chuyện sát nhau" và "ẩu đả".

### 🛠️ 5. `buoc2_dataloader.py` (Tiền xử lý Dữ liệu)
Chịu trách nhiệm load video, cắt ảnh, và thực hiện Data Augmentation (Tăng cường dữ liệu) như làm mờ, đổi độ tương phản, xoay lật ảnh để tăng tính đa dạng của dữ liệu trước khi đưa vào hàm loss để huấn luyện.

### 📦 6. `yolo26n.pt`
Trọng số của mô hình YOLOv8 Nano đã được tinh chỉnh, đóng vai trò như "con mắt" của hệ thống, chuyên phát hiện và theo dõi người cực nhanh làm đầu vào cho Pipeline.

---

## 6. Các điểm Tối ưu hóa Đặc biệt (Đã tích hợp)
- **Scale up Segments**: Tăng số lượng khung hình phân tích từ 8 lên 12 để "trí nhớ" mô hình dài hơn, hiểu các hành vi bạo lực phức tạp.
- **Detector Mode = Track**: Chuyển YOLO từ nhận diện khung hình rời rạc sang chế độ `track` liên tục, loại bỏ triệt để hiện tượng khung bao bị "rung bần bật" khi người đứng yên.
- **Hysteresis Logic**: Áp dụng các mốc `CONFIRM_ENTER_STREAK` và ngưỡng kép (`FIGHT_ON_THRESHOLD`, `FIGHT_OFF_THRESHOLD`) để cảnh báo không bị nháy liên tục khi xác suất nằm ở ngưỡng giới hạn.

---

## 7. Cấu hình Môi trường (Environment Profiles)
Hệ thống cho phép cấu hình theo ngữ cảnh cụ thể để tối ưu hóa khả năng phát hiện:
*   `balanced` (Mặc định): Cân bằng giữa độ nhạy và chống báo động giả.
*   `school_park`: Môi trường trường học/công viên. Ngưỡng cảnh báo cao hơn, cần nhiều thời gian tương tác (12 frames) để kích hoạt, chống báo động giả khi học sinh đùa giỡn.
*   `high_risk`: Môi trường an ninh cao (VD: nhà tù, ngõ hẻm đêm). Ngưỡng cảnh báo thấp, phát hiện ngay lập tức (chỉ cần 7 frames), chấp nhận hi sinh một chút độ chính xác để không bỏ lọt.

## 8. Đánh giá Mô hình & Dữ liệu (Evaluation & Datasets)
*   **Dữ liệu huấn luyện**: Hệ thống được huấn luyện trên bộ dữ liệu chuẩn **RWF-2000** và tinh chỉnh (fine-tune) sâu trên bộ dữ liệu **Data_Violence_Detection (Large)** do chúng ta tự xây dựng.
*   **Kết quả (Metrics)**: Phiên bản **MobileNetV3-Large (12 segments)** đạt độ chính xác cao nhất (**86.43% Accuracy** trên tập Validation), với chỉ số Recall vượt trội, đảm bảo tỷ lệ bỏ sót hành vi bạo lực (False Negatives) ở mức cực thấp. Cấu hình độ phân giải tối ưu được xác định là `imgsz=416`.

## 9. Mục tiêu Triển khai (Target Deployment)
Mã nguồn này được thiết kế không chỉ để chạy trên máy tính cấu hình cao (như RTX 3050) mà còn được chuẩn bị kiến trúc để xuất sang **ONNX** và **OpenVINO**. 
*   **Mục tiêu**: Đưa toàn bộ Pipeline (YOLO + TSM) chạy mượt mà theo thời gian thực (8-12 FPS) trên thiết bị nhúng (Edge Device) như **Raspberry Pi 4**.
*   **Tối ưu hóa**: Pipeline đã được gỡ bỏ các phụ thuộc không cần thiết và chuẩn bị để thay thế Backend PyTorch bằng OpenVINO Runtime, cho phép tiết kiệm tối đa RAM và CPU trên thiết bị phần cứng hạn chế.
