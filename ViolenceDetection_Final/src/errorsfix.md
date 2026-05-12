# Giải pháp Khắc phục Lỗi (Error Fix Log)
**Ngày sửa**: 2026-05-09
**Liên kết**: Tham chiếu từ `errors.md`

Tài liệu này ghi lại các phương pháp và định hướng sửa mã để giải quyết các vấn đề đã quan sát thấy trong hệ thống, làm nền tảng tri thức cho các phiên AI tiếp theo.

## 1. Sửa Lỗi 1: Bỏ sót hành vi do YOLO Tracking bị mất (False Negative)

**Phân tích cốt lõi**: Trong lúc xô xát, các đối tượng che khuất nhau hoặc ngã xuống đất khiến YOLO bị "mù" (det=0). Do Pipeline của chúng ta đang lập trình kiểu: "Chỉ khi nào Track_ID tồn tại thì mới đưa khung hình vào bộ nhớ TSM", nên khi YOLO mất tích, TSM cũng bị ngắt quãng theo, làm chuỗi 12 frames bị hỏng.

**Phương án khắc phục (Đã đề xuất cho `buoc6_main_pipeline.py`)**:
*   **Fallback to Full-Frame**: Cần lập trình thêm logic: Ngay cả khi YOLO không track được ai (số lượng đối tượng = 0), nếu trước đó đang có sự kiện hoặc đang nghi ngờ, hãy **cắt toàn bộ khung hình (Full Frame)** và tống thẳng vào mạng TSM để duy trì chuỗi 12 segments. Đừng để đứt đoạn dữ liệu vào TSM chỉ vì YOLO bị mất dấu.
*   **Giảm ngưỡng Yolo Confidence (`det_conf`)**: Trong các khung hình mờ hoặc hành động nhanh, YOLO có thể bị rớt Confidence. Cần cho phép hạ `det_conf` (VD: từ 0.35 xuống 0.15) để cố gắng "vớt" lại các bounding box bị mờ trong lúc đánh nhau.
*   **Giữ ID lâu hơn (Track Buffer)**: Cấu hình lại ByteTrack trong YOLO (tham số `track_buffer`) để nó nhớ mặt người lâu hơn (VD: 60-90 frames), kể cả khi người đó biến mất tạm thời dưới sàn nhà, tránh việc reset lại Tube từ đầu.
*   **Tăng độ nhạy TSM-only (Top-down)**: Giảm `--tsm-only-on-threshold` xuống khoảng `0.50` (off `0.36`) và dùng thêm `fused_prob`/`raw_fight_prob` trong `tsm_only` để bắt sớm các pha ra đòn khi góc quay từ trên xuống.

## 2. Sửa Lỗi 2: Khung ROI chưa bao quát đủ

**Phân tích cốt lõi**: Khi hai người đánh nhau, họ vung tay, vung chân rộng ra ngoài cái khung Bounding Box chữ nhật ôm sát người của YOLO. Nếu ta lấy khung quá sát, tay và chân đang đấm đá sẽ bị cắt mất.

**Phương án khắc phục (Đã đề xuất cho `buoc6_main_pipeline.py`)**:
*   **Tăng ROI_EXPAND_RATIO**: Hiện tại tỷ lệ mở rộng ROI có thể đang thấp. Cần tăng cường mở rộng ra xung quanh (VD: `ROI_EXPAND_RATIO = 1.8` đến `2.0`).
*   **Ưu tiên ghép nhóm (Merge Clusters)**: Tinh chỉnh lại thuật toán phân cụm (Clustering) khi xác định ai đang đánh ai. Thay vì tính khoảng cách tâm (Center Distance), hãy tính cả hình chữ nhật bao quanh lớn nhất (Bounding Box Union) của tất cả những người đang đứng gần nhau, đảm bảo cái "Tube" màu vàng bắt được toàn cảnh cuộc chiến.

## 3. Sửa Lỗi 3: FileNotFoundError khi chạy giao diện WinForm

**Phân tích cốt lõi**: Lỗi này là do sự thay đổi cấu trúc thư mục từ lúc phát triển sang lúc đóng gói (`ViolenceDetection_Final`). File script không tìm thấy model trọng số nằm ở đường dẫn gốc.

**Phương án khắc phục**:
*   **Đồng bộ Thư mục**: Tạo thư mục `weights/` bên trong `ViolenceDetection_Final` và di chuyển/sao chép tệp `best_tsm_topdown.pth` vào đúng vị trí này.
*   **Cấu hình Đường dẫn Tuyệt đối trong App**: Cập nhật `winform_app_v2.py` dùng `Path(__file__).resolve().parent` để lấy thư mục hiện tại của app làm gốc, đảm bảo không bao giờ bị lỗi sai đường dẫn (relative path error) bất kể người dùng khởi chạy file từ thư mục nào trên cmd.

## 4. Sửa Lỗi 4: Weights không khớp kiến trúc TSM (variant Small vs Large)

**Phân tích cốt lõi**: Weights `best_tsm_topdown.pth` được train với MobileNetV3 **Large**, trong khi pipeline đang khởi tạo `MobileNetV3_TSM` mặc định `variant="small"`. Do đó `load_state_dict` thất bại với Missing/Unexpected keys.

**Phương án khắc phục**:
*   **Bổ sung tham số `--tsm-variant`**: Thêm lựa chọn `small/large` và đặt mặc định `large` ở pipeline.
*   **Khởi tạo đúng kiến trúc**: Tạo model `MobileNetV3_TSM(..., variant=args.tsm_variant)` trước khi `load_state_dict`.
*   **UI truyền rõ ràng**: Trong `winform_app_v2.py`, truyền `--tsm-variant large` để tránh lệch kiến trúc khi chạy qua giao diện.

## 5. Sửa Lỗi 5: Camera/Stream bị check sai như file path

**Phân tích cốt lõi**: `--video-path` đang parse thành `Path`, nên chuỗi `"0"` hoặc URL bị hiểu là file và fail ở bước `exists()`.

**Phương án khắc phục**:
*   **Parse theo chuỗi**: Đổi `--video-path` thành kiểu `str`.
*   **Resolve nguồn video**: Nếu chuỗi chỉ gồm số -> cast `int` cho camera; nếu là `http/https/rtsp/rtmp` -> dùng trực tiếp; nếu không thì treat là file `Path` và mới kiểm tra `exists()`.
*   **Mở video đúng kiểu**: Gọi `cv2.VideoCapture(int)` cho camera, `cv2.VideoCapture(str(path))` cho file/URL.

## 6. Sửa Lỗi 6: KeyboardInterrupt khi chạy WinForm từ terminal

**Phân tích cốt lõi**: Khi người dùng bấm Ctrl+C để dừng app trong terminal, Python ném `KeyboardInterrupt` và in traceback.

**Phương án khắc phục**:
*   **Bắt `KeyboardInterrupt`** trong `__main__` của `winform_app_v2.py` để thoát êm và không in traceback.
*   **Cách chạy đề xuất**: Dùng `pythonw winform_app_v2.py` hoặc đóng cửa sổ UI thay vì Ctrl+C.

## 7. Sửa Lỗi 7: Camera xa không lên khung vàng (far-camera FN)

**Phân tích cốt lõi**: Người trong khung quá nhỏ nên bị lọc bởi ngưỡng kích thước và các ngưỡng motion/activation cao cho chuyển động nhỏ.

**Phương án khắc phục**:
*   **Thêm profile `far_camera`**: Nới lỏng `HUMAN_MIN_HEIGHT_RATIO`, `HUMAN_MIN_AREA_RATIO`, và giảm các ngưỡng activation/motion.
*   **Giảm `det_conf` mặc định khi dùng `far_camera`**: Hạ xuống ~0.12 để YOLO bắt được người nhỏ.
*   **Tăng `imgsz` khi chạy xa**: Khuyến nghị `imgsz=640` hoặc `768` để tăng khả năng phát hiện.
*   **Giảm ngưỡng TSM-only khi `far_camera`**: Hạ `tsm_only_on/off` xuống khoảng `0.42/0.30` nếu người nhỏ làm TSM raw thấp.
*   **Mở khóa fullframe-rescue sớm**: Đặt `FULLFRAME_RESCUE_MIN_SINGLE_GATE_STREAK=0` và tăng `FULLFRAME_RESCUE_BLEND` để rescue hiệu quả khi lock yếu.

## 8. Thử nghiệm Mới: Tắt hoàn toàn Khung Vàng (No-Tracking Mode)

**Phân tích cốt lõi**: Việc vẽ khung vàng (ROI) quá phụ thuộc vào Bounding Box của YOLO. Nếu YOLO vẽ sai hoặc bỏ sót một người lúc ngã, ROI sẽ cắt trượt mất khung hình bạo lực. 

**Phương án khắc phục (Đã triển khai)**:
*   Tạo bản **`buoc6_main_pipeline_notracking.py`**.
*   Bỏ lệnh vẽ `cv2.rectangle` màu vàng/cam/đỏ trên khung hình.
*   Bỏ luôn việc cắt xén khung hình `crop_img = raw_frame[y1:y2, x1:x2]`. Thay vào đó, nạp thẳng toàn bộ khung hình `full_img` vào mô hình TSM.
*   Tích hợp vào giao diện `winform_app_v2.py` bằng CheckBox: **"Tắt khung vàng (Chỉ Full-frame)"**.

## Lời nhắc cho AI (Self-Reflection Note)
Mỗi khi khởi động lại, AI cần đọc file này để nhớ rằng: **Sự hoàn hảo của hệ thống hiện tại bị giới hạn bởi khả năng Tracking của YOLO trong các pha va chạm mạnh**. Nếu thấy tỷ lệ Alert quá thấp, việc đầu tiên cần làm là kiểm tra xem tính năng Full-frame Rescue (Cứu nguy khung hình tổng) đã được trigger đúng cách khi Track_ID bị mất hay chưa, trước khi đụng vào các thông số Threshold. Đồng thời, luôn kiểm tra cấu trúc thư mục trước khi chạy Inference để tránh lỗi lặt vặt. Mọi thử nghiệm về việc tắt Tracking/ROI đều phải đối chiếu với độ chính xác trên Full-frame.
