import os
import shutil
import random

# ==============================
# CẤU HÌNH
image_folder = "D:\\VS code\\thigiacmaytinh\\BTL\\mask-seg\\dataset\\images"
label_folder = "D:\\VS code\\thigiacmaytinh\\BTL\\mask-seg\\dataset\\labels"

output_dir = "dataset"
images_output = os.path.join(output_dir, "images")
labels_output = os.path.join(output_dir, "labels")

train_ratio = 0.8
# ==============================

for split in ["train", "val"]:
    os.makedirs(os.path.join(images_output, split), exist_ok=True)
    os.makedirs(os.path.join(labels_output, split), exist_ok=True)

image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
random.shuffle(image_files)

train_count = int(len(image_files) * train_ratio)
train_files = image_files[:train_count]
val_files = image_files[train_count:]

def process_split(split_files, split_name):
    total = 0
    missing = 0

    for img_file in split_files:
        base_name = os.path.splitext(img_file)[0]  # frame_0030
        label_file = None

        for f in os.listdir(label_folder):
            if f.endswith(".txt") and base_name in f:
                label_file = f
                break

        if label_file is None:
            print(f"⚠️ Không tìm thấy nhãn cho {img_file}, bỏ qua")
            missing += 1
            continue

        new_label_name = base_name + ".txt"

        shutil.copyfile(
            os.path.join(image_folder, img_file),
            os.path.join(images_output, split_name, img_file)
        )

        shutil.copyfile(
            os.path.join(label_folder, label_file),
            os.path.join(labels_output, split_name, new_label_name)
        )

        print(f"✅ {split_name}: {img_file} → {new_label_name}")
        total += 1

    print(f"\n📊 {split_name.upper()} tổng cộng: {total} ảnh, {missing} ảnh thiếu nhãn")

process_split(train_files, "train")
process_split(val_files, "val")

print("\n🎉 Hoàn tất! Dataset đã chia tại thư mục 'dataset/'")
