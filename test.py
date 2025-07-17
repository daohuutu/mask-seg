import cv2
from ultralytics import YOLO

# Load mô hình segmentation đã huấn luyện
model = YOLO("runs/segment/train5/weights/best.pt")

# Mở video đầu vào
video_path = "D:\\Images\\War Thunder\\War Thunder 2025.06.30 - 16.10.07.01.mp4"
cap = cv2.VideoCapture(video_path)

# Lấy thông tin video gốc
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Tạo video ghi ra kết quả
output_path = "D:\\VS code\\thigiacmaytinh\\BTL\\mask-seg\\outputtest"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán segmentation
    results = model.predict(frame, conf=0.5, iou=0.5, task='segment')

    # Vẽ mask + nhãn lên khung hình
    annotated_frame = results[0].plot()

    # Hiển thị (tuỳ chọn)
    cv2.imshow("YOLOv8 Segmentation", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Đã lưu kết quả ra: {output_path}")
