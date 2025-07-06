import numpy as np
import cv2 as cv

# Load dữ liệu hiệu chuẩn từ file calib_data.npz
calib_data = np.load('calib_data.npz')
mtx = calib_data['mtx']          # Camera matrix (ma trận nội tại)
dist = calib_data['dist']        # Distortion coefficients (hệ số méo)

print("Camera Matrix:")
print(mtx)
print("\nDistortion Coefficients:")
print(dist)

# Khởi tạo camera
cap = cv.VideoCapture(0)

# Kiểm tra xem camera có mở được không
if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

# Lấy kích thước frame từ camera
ret, frame = cap.read()
if ret:
    h, w = frame.shape[:2]
    print(f"Kích thước frame: {w}x{h}")
    
    # Tính toán new camera matrix cho undistortion
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.5, (w, h))
    
    print("New Camera Matrix:")
    print(newcameramtx)
    print("ROI:", roi)
else:
    print("Không thể đọc frame từ camera!")
    cap.release()
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Không thể đọc frame!")
        break
    
    # Hiển thị ảnh gốc (bị méo)
    cv.imshow('Original (Distorted)', frame)
    
    # Áp dụng undistortion để loại bỏ méo
    undistorted = cv.undistort(frame, mtx, dist, None, newcameramtx)
    
    # Crop theo ROI nếu cần
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        undistorted_cropped = undistorted[y:y+h_roi, x:x+w_roi]
        cv.imshow('Undistorted (Cropped)', undistorted_cropped)
    
    # Hiển thị ảnh đã hiệu chỉnh (không crop)
    cv.imshow('Undistorted (Full)', undistorted)
    
    # Xử lý phím bấm
    key = cv.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv.destroyAllWindows()