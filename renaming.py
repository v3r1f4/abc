import os

# Thay đường dẫn này thành đường dẫn thư mục chứa ảnh của bạn
folder_path = r'D:\abc\samples'

# Lấy danh sách file trong thư mục và lọc ra các file ảnh
image_extensions = ('.jpg')
images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
images.sort()  # Sắp xếp để thứ tự tên file ổn định

for idx, filename in enumerate(images, 1):
    ext = os.path.splitext(filename)[1]  # Lấy phần mở rộng
    new_name = f"{idx}{ext}"
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    os.rename(src, dst)
    print(f"Renamed {filename} -> {new_name}")

print("Đã đổi tên xong!")