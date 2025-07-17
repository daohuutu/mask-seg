import cv2
import os

video_path = "C:\\Users\\Admin\\Videos\\Captures\\War Thunder - Lái thử 2025-06-28 16-45-37.mp4"
output_folder = "D:\\anhbtl"
os.makedirs(output_folder, exist_ok=True)

# Mở video
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)                 # số khung hình mỗi giây
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps                    # thời lượng video tính bằng giây

interval_seconds = 3                          # mỗi n giây lấy 1 ảnh
frame_interval = int(fps * interval_seconds)    # số frame giữa 2 lần trích ảnh

frame_id = 0
saved_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        filename = os.path.join(output_folder, f"frame_{saved_id:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_id += 1

    frame_id += 1

cap.release()
print(f"Đã trích {saved_id} ảnh từ video mỗi 10 giây vào thư mục '{output_folder}'")
