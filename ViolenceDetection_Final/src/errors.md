# Báo cáo Lỗi và Các vấn đề Quan sát được (Error Log)
**Ngày ghi nhận**: 2026-05-09
**Nguồn phân tích**: File video `demo_lockon_widefix.mp4`

Dựa trên việc quan sát đầu ra của Pipeline trong quá trình hoạt động, dưới đây là các phân tích chi tiết về trạng thái hiện tại của hệ thống:

## 1. Trạng thái chung (General Status)
*   Video gốc mang tính chất đánh nhau phức tạp trong một phòng đông người (có va chạm, xô xát, ngã xuống sàn).
*   Thống kê từ console: `Stats: frames=241 alert=0 lock=0 single_lock=0 single_ratio=0.0000.208`
*   Hệ thống chạy mượt, giao diện đã xuất hiện Letterbox Resize (giữ nguyên tỷ lệ), không còn bị méo. Khung bao (Bounding box) tracking màu vàng hoạt động ổn định.

## 2. Các lỗi quan sát được (Observed Issues)

### Lỗi 1 (Nghiêm trọng): Bỏ sót hành vi bạo lực (False Negative / Alert = 0)
*   **Mô tả**: Trong video rõ ràng có diễn ra một cuộc ẩu đả nghiêm trọng (từ 00:01 đến 00:08, có người bị đánh ngã xuống đất). Tuy nhiên, cờ `alert` (Cảnh báo bạo lực) vẫn trả về 0. Khung chữ trên màn hình vẫn báo màu xanh lá (Normal) dù hành động rất mạnh bạo.
*   **Mức độ**: Nghiêm trọng (Critical). Hệ thống giám sát không thể bỏ lọt hành vi đánh nhau rõ ràng như vậy.
*   **Phân tích Nguyên nhân**:
    1.  **Vấn đề Tracking ID (YOLO ByteTrack)**: Tại các thời điểm xung đột mạnh (đặc biệt khi ngã xuống), YOLO bị mất dấu đối tượng (`det=00` hiển thị trên màn hình ở các giây 00:03 - 00:09). Khi không phát hiện được bounding box, hệ thống không có "Tube" (ống ROI) để đẩy vào TSM phân tích.
    2.  **Lock-on Thất bại**: Ở giây 00:02, hệ thống cố gắng tạo một ROI màu vàng (SEARCH/CROWD mode), nhưng sau đó ngay lập tức bị mất (`lock=0`). Do đó, hành động liên hoàn phía sau không được đưa vào bộ nhớ đệm (buffer) để mô hình TSM nhận diện.
    3.  **Full-frame Rescue chưa kích hoạt đúng lúc**: Mặc dù tính năng `fullframe_rescue` được bật (`fullframe_rescue=True`), có vẻ như nó không được kích hoạt khi mất Tracking, dẫn đến việc bỏ sót toàn bộ hành vi sau đó.

### Lỗi 2 (Trung bình): Khung ROI Lock-on vẫn chưa đủ bao quát
*   **Mô tả**: Tại giây 00:01 - 00:02, khi bắt đầu có va chạm, hệ thống có tạo một khung màu vàng (`MODE:search`). Tuy nhiên, khung này (tube) quá tập trung vào một số cá nhân bên trên mà không bắt trọn vẹn cả người đang có nguy cơ ngã.
*   **Mức độ**: Trung bình.
*   **Phân tích Nguyên nhân**: Cơ chế gộp Bounding box (Merge BBox) của YOLO có thể chưa tính toán khoảng lề (Margin) đủ rộng đối với các góc quay hẹp từ trên xuống (top-down), làm mất đi phần tay chân đang vung vẩy.

### Lỗi 3 (Giao diện / Khởi tạo): FileNotFoundError khi chạy qua WinForm
*   **Mô tả**: Bị văng lỗi `FileNotFoundError: TSM weights not found: C:\Users\Lenovo\Desktop\NewVioDe\ViolenceDetection_Final\weights\best_tsm_topdown.pth` khi nhấn nút chạy trên giao diện `winform_app_v2.py`.
*   **Mức độ**: Chặn đứng (Blocker). Không thể chạy được Pipeline.
*   **Phân tích Nguyên nhân**: Quá trình khởi tạo thư mục `ViolenceDetection_Final` đã thiếu mất thư mục `weights` và file trọng số `.pth` bên trong. Code trong `buoc6_main_pipeline.py` dùng `Path.exists()` để bắt buộc file phải có thực mới cho phép chạy tiếp, nên nó đã dừng chương trình ngay khi vừa đọc tham số.

### Lỗi 4 (Blocker): Không load được weights do sai kiến trúc TSM
*   **Mô tả**: Khi chạy WinForm, pipeline dừng với lỗi `RuntimeError: Error(s) in loading state_dict for MobileNetV3_TSM` kèm danh sách Missing/Unexpected keys và size mismatch.
*   **Mức độ**: Chặn đứng (Blocker). Pipeline không chạy được.
*   **Phân tích Nguyên nhân**: Model TSM đang khởi tạo mặc định theo `variant="small"` trong khi weights `best_tsm_topdown.pth` được train theo MobileNetV3 **Large** (12 segments). Kiến trúc không khớp nên `load_state_dict` thất bại.

### Lỗi 5 (Blocker cho Camera/Stream): Input video not found khi chọn Camera/DroidCam
*   **Mô tả**: Khi chọn camera (ID `0`) hoặc nguồn stream, pipeline báo lỗi `FileNotFoundError: Input video not found: 0` và dừng.
*   **Mức độ**: Blocker cho camera/stream.
*   **Phân tích Nguyên nhân**: Tham số `--video-path` được parse thành `Path`, nên chuỗi `"0"` hoặc URL bị hiểu như đường dẫn file và bị kiểm tra `exists()` trước khi mở `cv2.VideoCapture`.

### Quan sát Batch SCVD (Nghi ngờ FN): 3 video không có alert
*   **Mô tả**: Chạy batch trên `SCVD_converted/Test/Violence` (12 video) cho thấy 3 video có `alert_frames=0`.
*   **Danh sách**: `t_v004_converted.avi`, `t_v005_converted.avi`, `t_v008_converted.avi`.
*   **Mức độ**: Nghi ngờ False Negative (chưa xác minh bằng mắt). Cần kiểm tra output video để xác nhận.

### Lỗi 6 (Trải nghiệm): KeyboardInterrupt khi chạy WinForm từ terminal
*   **Mô tả**: Khi chạy `python winform_app_v2.py` trong terminal rồi bấm Ctrl+C (hoặc dừng thủ công), xuất hiện traceback `KeyboardInterrupt`.
*   **Mức độ**: Trải nghiệm (không ảnh hưởng chạy thực tế).
*   **Phân tích Nguyên nhân**: Tín hiệu Ctrl+C được Python ném thành `KeyboardInterrupt` khi app đang chạy loop.

### Lỗi 7 (Nhận diện xa): Camera xa không lên khung vàng, trạng thái luôn Normal
*   **Mô tả**: Khi đặt camera xa, khung lock-on màu vàng không xuất hiện và xác suất bạo lực không tăng.
*   **Mức độ**: Nghiêm trọng (FN). Không nhận diện được đánh nhau từ xa.
*   **Phân tích Nguyên nhân**: Bounding box người quá nhỏ nên bị lọc bởi ngưỡng `HUMAN_MIN_HEIGHT_RATIO` / `HUMAN_MIN_AREA_RATIO`, cộng thêm các ngưỡng activation/motion quá cao cho chuyển động nhỏ theo pixel.

## 3. Tổng kết
Mặc dù đã sửa được lỗi "nhảy khung" (jitter) và ổn định được giao diện hiển thị, Pipeline hiện tại đang bị phụ thuộc quá nhiều vào sự hoàn hảo của YOLO Tracking. Khi xảy ra ẩu đả mạnh (người đan xen vào nhau, ngã xuống sàn), YOLO bị mù tạm thời khiến TSM không có dữ liệu đầu vào, dẫn đến bỏ lọt tội phạm.

Cần một giải pháp sửa lỗi triệt để ở `buoc6_main_pipeline.py`.
