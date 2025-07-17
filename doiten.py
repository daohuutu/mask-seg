import os
import shutil

# Đường dẫn thư mục
label_folder = "D:\\VS code\\thigiacmaytinh\\BTL\\mask-seg\\tankmask\\labels"
new_label_folder = "D:\\VS code\\thigiacmaytinh\\BTL\\mask-seg\\tankmask\\labels"
os.makedirs(new_label_folder, exist_ok=True)

# Duyệt các file trong thư mục labels
for filename in os.listdir(label_folder):
    if filename.endswith(".txt"):
        # Tách phần frame_XXXX
        parts = filename.split("frame_")
        if len(parts) == 2:
            new_name = f"frame_{parts[1]}"
            old_path = os.path.join(label_folder, filename)
            new_path = os.path.join(new_label_folder, new_name)
            shutil.copyfile(old_path, new_path)

print("✅ Đã đổi tên xong! File nhãn mới nằm trong 'labels_fixed/'")
