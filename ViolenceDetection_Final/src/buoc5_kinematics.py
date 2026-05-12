import math
from collections import defaultdict

class KinematicsGate:
    def __init__(self, history_size=5, v_threshold=10.0, a_threshold=5.0):
        self.history_size = history_size
        self.v_threshold = v_threshold
        self.a_threshold = a_threshold
        self.track_history = defaultdict(list)

    def update_and_check(self, track_boxes):
        suspicious_ids = set()
        current_ids = set()

        for box in track_boxes:
            track_id, x1, y1, x2, y2 = box
            current_ids.add(track_id)
            
            # Tính tọa độ tâm
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1

            history = self.track_history[track_id]
            history.append((cx, cy, w, h))

            if len(history) > self.history_size:
                history.pop(0)

            if len(history) >= 3:
                p1, p2, p3 = history[-3], history[-2], history[-1]
                
                # 🌟 TÍNH TOÁN VECTOR DI CHUYỂN
                dx = p3[0] - p2[0] # Chuyển động ngang
                dy = p3[1] - p2[1] # Chuyển động dọc (Dương là đi xuống)
                
                v1 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                v2 = math.hypot(dx, dy)
                
                acceleration = abs(v2 - v1)
                relative_v = v2 / max(p3[3], 1.0) 
                
                # 🌟 BỘ LỌC VECTOR ĐỈNH CAO: CHỐNG NHIỄU CÚI LẠY / NGỒI XỔM
                # Nếu hướng chuyển động là đi xuống đất (dy > 0) 
                # VÀ chiều dọc áp đảo hoàn toàn chiều ngang (dy > |dx| * 2.0)
                if dy > 0 and dy > abs(dx) * 2.0:
                    continue # Bỏ qua ngay lập tức! Đây không phải đòn đấm/đá.
                
                # Nếu vượt qua bộ lọc Vector, mới xét vận tốc và gia tốc
                if v2 > self.v_threshold or relative_v > 0.08 or acceleration > self.a_threshold:
                    suspicious_ids.add(track_id)
        
        # Dọn dẹp memory
        active_ids = list(self.track_history.keys())
        for tid in active_ids:
            if tid not in current_ids:
                del self.track_history[tid]
                
        return suspicious_ids