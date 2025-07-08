import cv2
import numpy as np
image = cv2.imread('ArUco/11.jpg')  # Ảnh có marker
marker_length_m = 0.09    # Ví dụ: marker dài 5 cm

calib_data = np.load('calib_data.npz')
camera_matrix = calib_data['mtx']
dist_coeffs = calib_data['dist']

def compute_pixel_to_meter_ratio(image, marker_length_meters, camera_matrix, dist_coeffs):
    # Khởi tạo detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Chuyển xám và phát hiện marker
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or len(corners) == 0:
        print("❌ Không tìm thấy marker trong ảnh.")
        return None

    # Giả sử lấy marker đầu tiên tìm được
    corner_points = corners[0][0]  # 4 điểm góc

    # Tính chiều dài trung bình giữa các cạnh marker trong pixel
    side_lengths = []
    for i in range(4):
        pt1 = corner_points[i]
        pt2 = corner_points[(i + 1) % 4]
        side_length = np.linalg.norm(pt1 - pt2)
        side_lengths.append(side_length)

    avg_pixel_length = np.mean(side_lengths)

    # Tính tỉ lệ
    pixel_to_meter_ratio = avg_pixel_length / marker_length_meters

    print(f"✓ Trung bình {avg_pixel_length:.2f} pixels ↔ {marker_length_meters} meters")
    print(f"→ Tỉ lệ tính được: {pixel_to_meter_ratio:.2f} pixels/meter")

    return pixel_to_meter_ratio

ratio = compute_pixel_to_meter_ratio(image, marker_length_m, camera_matrix, dist_coeffs)
