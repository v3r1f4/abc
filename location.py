import cv2
import numpy as np
from scipy.spatial.transform import Rotation

class ArUcoImageProcessor:
    def __init__(self, camera_matrix, dist_coeffs, marker_length):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.origin_pixel = None
        self.pixel_to_meter_ratio = 1 #7504.89

    def detect_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            return corners, ids, rvecs, tvecs
        return None, None, None, None

    def draw_coordinate_system(self, image, rvec, tvec, length=0.03):
        axis_points = np.array([
            [0, 0, 0],
            [length, 0, 0],
            [0, length, 0],
            [0, 0, -length]
        ], dtype=np.float32)

        projected_points, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)

        origin = tuple(projected_points[0].ravel().astype(int))
        x_axis = tuple(projected_points[1].ravel().astype(int))
        y_axis = tuple(projected_points[2].ravel().astype(int))
        z_axis = tuple(projected_points[3].ravel().astype(int))

        cv2.arrowedLine(image, origin, x_axis, (0, 0, 255), 3)
        cv2.arrowedLine(image, origin, y_axis, (0, 255, 0), 3)
        cv2.arrowedLine(image, origin, z_axis, (255, 0, 0), 3)

        return image

    def get_marker_coordinates(self, rvec, tvec):
        position_camera = tvec[0]  # Với estimatePoseSingleMarkers, tvec có shape (1, 1, 3)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        rotation = Rotation.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        return position_camera, euler_angles

    def pixel_to_world_coordinates(self, pixel_point):
        if self.origin_pixel is None:
            return None

        dx_pixel = pixel_point[0] - self.origin_pixel[0]
        dy_pixel = pixel_point[1] - self.origin_pixel[1]

        x_world = dx_pixel/self.pixel_to_meter_ratio
        y_world = dy_pixel/self.pixel_to_meter_ratio  # Không âm vì gốc ở góc trên trái

        return (x_world, y_world)

    def draw_origin_coordinate_system(self, image):
        if self.origin_pixel is None:
            return image

        cv2.circle(image, self.origin_pixel, 8, (0, 0, 255), -1)

        axis_length = 60
        x_end = (self.origin_pixel[0] + axis_length, self.origin_pixel[1])
        cv2.arrowedLine(image, self.origin_pixel, x_end, (0, 0, 255), 3)
        cv2.putText(image, 'X', (x_end[0] + 5, x_end[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        y_end = (self.origin_pixel[0], self.origin_pixel[1] + axis_length)  # Y hướng xuống
        cv2.arrowedLine(image, self.origin_pixel, y_end, (0, 255, 0), 3)
        cv2.putText(image, 'Y', (y_end[0] + 5, y_end[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(image, 'ORIGIN (0,0)', 
                   (self.origin_pixel[0] + 15, self.origin_pixel[1] - 25),  # Hiển thị phía trên gốc
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return image

    def process_image(self, image, set_origin=True):
        if set_origin and self.origin_pixel is None:
            self.set_origin_by_click(image)

        corners, ids, rvecs, tvecs = self.detect_markers(image)

        result_image = image.copy()
        cv2.aruco.drawDetectedMarkers(result_image, corners, ids)
        result_image = self.draw_origin_coordinate_system(result_image)

        marker_info = {}

        origin_tvec = None
        if len(tvecs) > 0:
            origin_tvec = tvecs[0][0] 

        for i in range(len(ids)):
            marker_id = ids[i][0]
            rvec = rvecs[i]
            tvec = tvecs[i]

            result_image = self.draw_coordinate_system(result_image, rvec, tvec)
            position_camera, euler_angles = self.get_marker_coordinates(rvec, tvec)
            corner_points = corners[i][0]
            center_pixel = np.mean(corner_points, axis=0)
            
            # Tính tọa độ tương đối so với origin (nếu có origin_pixel thì dùng pixel, nếu không thì dùng marker đầu tiên)
            if self.origin_pixel is not None:
                # Sử dụng pixel-to-world conversion với tỷ lệ phù hợp
                # Ước tính tỷ lệ từ khoảng cách thực và khoảng cách pixel
                if len(tvecs) >= 2:
                    # Tính tỷ lệ pixel/meter từ 2 marker đầu tiên
                    tvec1 = tvecs[0][0]
                    tvec2 = tvecs[1][0]
                    real_distance = np.linalg.norm(tvec1 - tvec2)
                    
                    corners1 = corners[0][0]
                    corners2 = corners[1][0]
                    center1 = np.mean(corners1, axis=0)
                    center2 = np.mean(corners2, axis=0)
                    pixel_distance = np.linalg.norm(center1 - center2)
                    
                    if pixel_distance > 0:
                        self.pixel_to_meter_ratio = pixel_distance / real_distance
                
                world_2d = self.pixel_to_world_coordinates(center_pixel)
            else:
                # Nếu không có origin_pixel, dùng marker đầu tiên làm gốc
                if origin_tvec is not None:
                    marker_tvec = tvec[0]
                    relative_pos = marker_tvec - origin_tvec
                    world_2d = (relative_pos[0], relative_pos[1])
                else:
                    world_2d = (position_camera[0], position_camera[1])

            text_pos = (int(center_pixel[0]) + 10, int(center_pixel[1]) - 10)
            cv2.putText(result_image, f"ID: {marker_id}",
                       text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            text_pos = (int(center_pixel[0]) + 10, int(center_pixel[1]) + 10)
            cv2.putText(result_image, f"3D: ({position_camera[0]:.3f}, {position_camera[1]:.3f}, {position_camera[2]:.3f})", 
                       text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if world_2d is not None:
                text_pos = (int(center_pixel[0]) + 10, int(center_pixel[1]) + 25)
                cv2.putText(result_image, f"2D: ({world_2d[0]:.3f}, {world_2d[1]:.3f})", 
                           text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            marker_info[marker_id] = {
                'position_3d_camera': position_camera,
                'position_2d_world': world_2d,  # mét
                'rotation_euler': euler_angles,
                'pixel_center': center_pixel
            }

        return result_image, marker_info

def main():
    calib_data = np.load('calib_data.npz')
    camera_matrix = calib_data['mtx']
    dist_coeffs = calib_data['dist']
    
    marker_length = 0.13  # 13cm
    
    # Khởi tạo processor
    processor = ArUcoImageProcessor(camera_matrix, dist_coeffs, marker_length)
    
    # Khởi tạo camera
    cap = cv2.VideoCapture(0)  # Camera mặc định
    
    # Đặt độ phân giải camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Đặt gốc tọa độ mặc định ở góc trên trái
    first_frame = True
    
    while True:
        _, frame = cap.read()
        
        # Đặt gốc tọa độ ở góc trên trái lần đầu tiên
        if first_frame:
            processor.origin_pixel = (20, 20)  # Góc trên trái với offset nhỏ
            first_frame = False
        
        # Xử lý frame (không cần set origin mỗi lần)
        result_image, marker_info = processor.process_image(frame, set_origin=False)
        
        # In thông tin markers (chỉ khi có marker)
        if marker_info:
            print("\r" + "="*60, end="")
            print(f"\rMarkers detected: {len(marker_info)} | ", end="")
            for marker_id, info in marker_info.items():
                if info['position_2d_world'] is not None:
                    x, y = info['position_2d_world']
                    print(f"ID{marker_id}:({x:.3f},{y:.3f})m ", end="")
            print("", end="", flush=True)
        
        # Hiển thị kết quả
        cv2.imshow('ArUco Real-time Detection', result_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()