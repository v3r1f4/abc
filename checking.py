import cv2
import numpy as np

# Giả sử bạn đã load được camera_matrix, dist_coeffs từ bước calib
# camera_matrix = ...
# dist_coeffs = ...
calib_data = np.load('calib_data_2.npz')
camera_matrix = calib_data['mtx']
dist_coeffs = calib_data['dist']

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Đọc ảnh từ camera hoặc file
frame = cv2.imread('Aruco/checking_2.jpg')

corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

if ids is not None and len(ids) >= 2:
    markerLength = 0.13  # chiều dài cạnh marker (m), thay đổi cho đúng kích thước thực tế
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist_coeffs)
    
    # Ví dụ: đo khoảng cách giữa 2 marker đầu tiên
    tvec1 = tvecs[0][0]
    tvec2 = tvecs[1][0]
    
    distance = np.linalg.norm(tvec1 - tvec2)
    print("Khoảng cách giữa 2 ArUco marker là:", distance, "m")
else:
    print("Không tìm thấy đủ marker trong ảnh")