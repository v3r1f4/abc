import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
import threading
import queue
import time
import serial
import serial.tools.list_ports

class ArUcoImageProcessor:
    def __init__(self, camera_matrix, dist_coeffs, marker_length):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.origin_pixel = None
        self.pixel_to_meter_ratio = 1

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
        position_camera = tvec[0]
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        rotation = Rotation.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        print(position_camera, euler_angles)
        return position_camera, euler_angles

    def pixel_to_world_coordinates(self, pixel_point):
        if self.origin_pixel is None:
            return None

        dx_pixel = pixel_point[0] - self.origin_pixel[0]
        dy_pixel = pixel_point[1] - self.origin_pixel[1]

        x_world = dx_pixel/self.pixel_to_meter_ratio
        y_world = dy_pixel/self.pixel_to_meter_ratio

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

        y_end = (self.origin_pixel[0], self.origin_pixel[1] + axis_length)
        cv2.arrowedLine(image, self.origin_pixel, y_end, (0, 255, 0), 3)
        cv2.putText(image, 'Y', (y_end[0] + 5, y_end[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(image, 'ORIGIN (0,0)', 
                   (self.origin_pixel[0] + 15, self.origin_pixel[1] - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return image

    def process_image(self, image, set_origin=True):
        corners, ids, rvecs, tvecs = self.detect_markers(image)

        result_image = image.copy()
        cv2.aruco.drawDetectedMarkers(result_image, corners, ids)
        result_image = self.draw_origin_coordinate_system(result_image)

        marker_info = {}

        if ids is not None and tvecs is not None and len(tvecs) > 0:
            origin_tvec = tvecs[0][0]

            for i in range(len(ids)):
                marker_id = ids[i][0]
                rvec = rvecs[i]
                tvec = tvecs[i]

                result_image = self.draw_coordinate_system(result_image, rvec, tvec)
                position_camera, euler_angles = self.get_marker_coordinates(rvec, tvec)
                corner_points = corners[i][0]
                center_pixel = np.mean(corner_points, axis=0)
                
                # Tính tọa độ tương đối
                if self.origin_pixel is not None:
                    if len(tvecs) >= 2:
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
                    marker_tvec = tvec[0]
                    relative_pos = marker_tvec - origin_tvec
                    world_2d = (relative_pos[0], relative_pos[1])

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
                    'position_2d_world': world_2d,
                    'rotation_euler': euler_angles,
                    'pixel_center': center_pixel
                }

        return result_image, marker_info


class CameraMappingCombined:
    def __init__(self):
        # Load camera calibration data
        calib_data = np.load('calib_data.npz')
        camera_matrix = calib_data['mtx']
        dist_coeffs = calib_data['dist']
        marker_length = 0.13  # 13cm
        
        # Initialize ArUco processor
        self.processor = ArUcoImageProcessor(camera_matrix, dist_coeffs, marker_length)
        
        # Robot visualization parameters
        self.radius = 0.05  # Giảm từ 0.25 xuống 0.05 để phù hợp với tỷ lệ mới
        self.lim = 1  # Giảm từ 5 xuống 1
        
        # Robot state - support multiple markers
        self.markers = {}  # Dictionary to store multiple markers: {id: {x, y, heading, trail_x, trail_y, last_seen}}
        self.robot_x = 0
        self.robot_y = 0
        self.robot_heading = 0
        
        # Marker timeout - remove markers not seen for this duration (seconds)
        self.marker_timeout = 0.1  # 2 giây
        
        # Add filtering for noise reduction
        self.filter_size = 5  # Số mẫu để tính trung bình
        self.marker_filters = {}  # Dictionary for each marker's filter history
        
        # Add trail for robot movement
        self.trail_x = []
        self.trail_y = []
        
        # Camera and threading
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.position_queue = queue.Queue()
        self.image_queue = queue.Queue()
        self.running = True
        
        # Set origin at top-left with small offset
        self.processor.origin_pixel = (20, 20)
        
        # Serial communication for sending to ESP32
        self.serial_connection = None
        self.init_serial_connection()
        
        # Command sending parameters
        self.last_command_time = {}  # Track last command time for each marker
        self.command_interval = 0.1  # Send commands every 100ms
        
    def init_serial_connection(self):
        """Khởi tạo kết nối serial với ESP32 sender"""
        try:
            # Tự động tìm cổng COM
            ports = serial.tools.list_ports.comports()
            for port in ports:
                if 'CH340' in port.description or 'CP210' in port.description or 'Arduino' in port.description:
                    try:
                        self.serial_connection = serial.Serial(port.device, 115200, timeout=1)
                        print(f"Đã kết nối với ESP32 sender qua {port.device}")
                        time.sleep(2)  # Chờ ESP32 khởi động
                        return
                    except:
                        continue
            
            # Nếu không tìm thấy, thử các cổng thông dụng
            common_ports = ['COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8']
            for port in common_ports:
                try:
                    self.serial_connection = serial.Serial(port, 115200, timeout=1)
                    print(f"Đã kết nối với ESP32 sender qua {port}")
                    time.sleep(2)
                    return
                except:
                    continue
                    
            print("Không thể kết nối với ESP32 sender")
        except Exception as e:
            print(f"Lỗi khi khởi tạo serial: {e}")
    
    def send_position_command(self, marker_id, x, y, heading):
        """Gửi lệnh vị trí đến ESP32 thông qua sender.ino"""
        if self.serial_connection is None:
            return
            
        current_time = time.time()
        
        # Kiểm tra interval để không gửi quá nhiều lệnh
        if marker_id in self.last_command_time:
            if current_time - self.last_command_time[marker_id] < self.command_interval:
                return
                
        try:
            # Format: "ID vx vy omega" 
            # Tạm thời gửi vị trí thay vì vận tốc
            # Có thể điều chỉnh để tính toán vận tốc từ sự thay đổi vị trí
            vx = x * 10  # Scale factor để chuyển từ m sang cm
            vy = y * 10
            omega = np.degrees(heading)  # Chuyển từ radian sang độ
            
            command = f"{marker_id} {vx:.2f} {vy:.2f} {omega:.2f}\n"
            self.serial_connection.write(command.encode())
            
            print(f"Sent to ESP32: {command.strip()}")
            self.last_command_time[marker_id] = current_time
            
        except Exception as e:
            print(f"Lỗi khi gửi lệnh serial: {e}")
            # Thử kết nối lại
            self.init_serial_connection()
        
    def apply_filter(self, marker_id, x, y, heading):
        """Áp dụng bộ lọc trung bình để giảm nhiễu cho từng marker riêng biệt"""
        # Khởi tạo filter history cho marker mới
        if marker_id not in self.marker_filters:
            self.marker_filters[marker_id] = {
                'x_history': [],
                'y_history': [],
                'heading_history': []
            }
        
        filter_data = self.marker_filters[marker_id]
        
        # Thêm giá trị mới vào lịch sử
        filter_data['x_history'].append(x)
        filter_data['y_history'].append(y)
        filter_data['heading_history'].append(heading)
        
        # Giữ chỉ filter_size mẫu gần nhất
        if len(filter_data['x_history']) > self.filter_size:
            filter_data['x_history'].pop(0)
            filter_data['y_history'].pop(0)
            filter_data['heading_history'].pop(0)
        
        # Tính trung bình
        filtered_x = np.mean(filter_data['x_history'])
        filtered_y = np.mean(filter_data['y_history'])
        
        # Đối với góc, cần xử lý đặc biệt do tính chất tuần hoàn
        cos_angles = [np.cos(h) for h in filter_data['heading_history']]
        sin_angles = [np.sin(h) for h in filter_data['heading_history']]
        filtered_heading = np.arctan2(np.mean(sin_angles), np.mean(cos_angles))
        
        return filtered_x, filtered_y, filtered_heading
    
    def camera_thread(self):
        """Thread để xử lý camera và phát hiện marker"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Cannot read camera frame")
                continue
                
            # Process frame to detect markers
            result_image, marker_info = self.processor.process_image(frame, set_origin=False)
            
            # Get current time for marker timeout tracking
            current_time = time.time()
            
            # Debug: Print all detected markers
            if marker_info:
                detected_ids = list(marker_info.keys())
                print(f"Detected markers: {detected_ids}")
                
                # Process all detected markers
                for robot_marker_id in detected_ids:
                    info = marker_info[robot_marker_id]
                    # Use 3D camera coordinates directly (more reliable)
                    position_3d = info['position_3d_camera']
                    x = position_3d[0]  # X from camera
                    y = position_3d[1]  # Y from camera
                    
                    # Calculate heading from rotation
                    euler_angles = info['rotation_euler']
                    heading = np.radians(euler_angles[2])  # Z rotation in radians
                    
                    print(f"Marker ID{robot_marker_id} at 3D({x:.3f}, {y:.3f}, {position_3d[2]:.3f}), heading: {np.degrees(heading):.1f}°")
                    
                    # Put new position in queue with marker ID and timestamp
                    try:
                        self.position_queue.put_nowait((robot_marker_id, x, y, heading, current_time))
                    except queue.Full:
                        pass
            else:
                print("No markers detected")
                
            # Send marker cleanup signal to remove old markers
            try:
                self.position_queue.put_nowait(("CLEANUP", current_time, None, None, None))
            except queue.Full:
                pass
            
            # Put processed image in queue for display
            try:
                # Resize image for display and convert BGR to RGB
                display_image = cv2.resize(result_image, (640, 480))
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                self.image_queue.put_nowait(display_image)
            except queue.Full:
                pass
            
            # Check for quit
            if not plt.get_fignums():  # If matplotlib window is closed
                self.running = False
                break
                
    def setup_plot(self):
        """Thiết lập matplotlib plot với camera và map cho multi-marker"""
        self.fig, (self.ax_camera, self.ax_map) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Setup camera subplot
        self.ax_camera.set_title("Camera View - ArUco Detection")
        self.ax_camera.axis('off')
        # Initialize empty image
        self.camera_image = self.ax_camera.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Setup map subplot for multi-marker support
        self.ax_map.set_aspect('equal')
        self.ax_map.set_xlim(-1, 1)
        self.ax_map.set_ylim(-1, 1)
        self.ax_map.set_xticks(np.arange(-1, 1.1, 0.2))
        self.ax_map.set_yticks(np.arange(-1, 1.1, 0.2))
        self.ax_map.axhline(0, color='black', zorder=1)
        self.ax_map.axvline(0, color='black', zorder=1)
        self.ax_map.grid(True, zorder=0)
        self.ax_map.set_xlabel('x (meters)')
        self.ax_map.set_ylabel('y (meters)')
        self.ax_map.set_title("Multi-ArUco Markers 2D Real-time Position")
        
    def init_animation(self):
        """Khởi tạo animation cho multi-marker"""
        return [self.camera_image]
        
    def update_animation(self, frame):
        """Cập nhật animation mỗi frame với hỗ trợ nhiều marker"""
        # Update camera view
        try:
            latest_image = None
            while True:
                latest_image = self.image_queue.get_nowait()
        except queue.Empty:
            pass
        
        if latest_image is not None:
            self.camera_image.set_array(latest_image)
        
        # Process all positions in queue
        position_updates = 0
        current_frame_time = time.time()
        
        while not self.position_queue.empty():
            try:
                data = self.position_queue.get_nowait()
                
                # Check if this is a cleanup signal
                if data[0] == "CLEANUP":
                    cleanup_time = data[1]
                    # Remove markers that haven't been seen for too long
                    markers_to_remove = []
                    for marker_id, marker_data in self.markers.items():
                        if cleanup_time - marker_data['last_seen'] > self.marker_timeout:
                            markers_to_remove.append(marker_id)
                    
                    for marker_id in markers_to_remove:
                        del self.markers[marker_id]
                        # Also remove filter data
                        if marker_id in self.marker_filters:
                            del self.marker_filters[marker_id]
                        print(f"REMOVED MARKER ID{marker_id} - not seen for {self.marker_timeout}s")
                    
                    continue
                
                # Normal marker position update
                marker_id, x, y, heading, timestamp = data
                position_updates += 1
                
                # Apply filter for this marker
                filtered_x, filtered_y, filtered_heading = self.apply_filter(marker_id, x, y, heading)
                
                # Update marker data
                if marker_id not in self.markers:
                    self.markers[marker_id] = {
                        'x': filtered_x, 'y': filtered_y, 'heading': filtered_heading,
                        'trail_x': [filtered_x], 'trail_y': [filtered_y],
                        'last_seen': timestamp
                    }
                    print(f"NEW MARKER ID{marker_id} created at ({filtered_x:.3f}, {filtered_y:.3f})")
                else:
                    self.markers[marker_id]['x'] = filtered_x
                    self.markers[marker_id]['y'] = filtered_y
                    self.markers[marker_id]['heading'] = filtered_heading
                    self.markers[marker_id]['last_seen'] = timestamp
                    
                    # Update trail only if position changed significantly
                    if (abs(filtered_x - self.markers[marker_id]['trail_x'][-1]) > 0.01 or 
                        abs(filtered_y - self.markers[marker_id]['trail_y'][-1]) > 0.01):
                        self.markers[marker_id]['trail_x'].append(filtered_x)
                        self.markers[marker_id]['trail_y'].append(filtered_y)
                        
                        # Limit trail length
                        if len(self.markers[marker_id]['trail_x']) > 100:
                            self.markers[marker_id]['trail_x'].pop(0)
                            self.markers[marker_id]['trail_y'].pop(0)
                
                # Send position to ESP32 via serial
                self.send_position_command(marker_id, filtered_x, filtered_y, filtered_heading)
                
                print(f"Marker ID{marker_id} - Raw: ({x:.3f}, {y:.3f}, {np.degrees(heading):.1f}°) -> Filtered: ({filtered_x:.3f}, {filtered_y:.3f}, {np.degrees(filtered_heading):.1f}°)")
                
            except queue.Empty:
                break
        
        if position_updates > 0:
            print(f"Animation frame {frame}: Updated {position_updates} markers, total markers: {len(self.markers)}")
        
        # Clear map and redraw with all markers
        self.ax_map.clear()
        
        # Setup map axes
        self.ax_map.set_aspect('equal')
        self.ax_map.set_xlim(-1, 1)
        self.ax_map.set_ylim(-1, 1)
        self.ax_map.set_xticks(np.arange(-1, 1.1, 0.2))
        self.ax_map.set_yticks(np.arange(-1, 1.1, 0.2))
        self.ax_map.axhline(0, color='black', zorder=1)
        self.ax_map.axvline(0, color='black', zorder=1)
        self.ax_map.grid(True, zorder=0)
        self.ax_map.set_xlabel('x (meters)')
        self.ax_map.set_ylabel('y (meters)')
        self.ax_map.set_title("Multi-ArUco Markers 2D Real-time Position")
        
        # Draw all markers
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        drawn_elements = []
        
        for i, (marker_id, marker_data) in enumerate(self.markers.items()):
            x, y, heading = marker_data['x'], marker_data['y'], marker_data['heading']
            trail_x, trail_y = marker_data['trail_x'], marker_data['trail_y']
            
            # Choose color for this marker
            color = colors[i % len(colors)]
            
            # Draw trail
            if len(trail_x) > 1:
                trail_line = self.ax_map.plot(trail_x, trail_y, color=color, alpha=0.5, linewidth=1, zorder=2, label=f'Trail ID{marker_id}')[0]
                drawn_elements.append(trail_line)
            
            # Draw robot circle
            circle = plt.Circle((x, y), 0.05, fill=False, color=color, linewidth=2, zorder=3)
            self.ax_map.add_patch(circle)
            drawn_elements.append(circle)
            
            # Draw heading line
            heading_length = 0.15
            x_end = x + heading_length * np.cos(heading)
            y_end = y + heading_length * np.sin(heading)
            heading_line = self.ax_map.plot([x, x_end], [y, y_end], color=color, linewidth=3, zorder=4, 
                           marker='>', markersize=8, markeredgecolor=color, markerfacecolor=color)[0]
            drawn_elements.append(heading_line)
            
            # Add position text
            text = self.ax_map.text(x + 0.1, y + 0.1, f'ID{marker_id}\n({x:.3f}, {y:.3f})\nAngle: {np.degrees(heading):.1f}°',
                           fontsize=8, color=color, verticalalignment='bottom', horizontalalignment='left',
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1, alpha=0.8),
                           zorder=5)
            drawn_elements.append(text)
        
        # Show legend if there are markers
        if self.markers:
            legend = self.ax_map.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=8)
            drawn_elements.append(legend)
        
        # Force redraw
        self.fig.canvas.draw_idle()
        
        return [self.camera_image] + drawn_elements
        
    def run(self):
        """Chạy chương trình chính"""
        # Start camera thread
        camera_thread = threading.Thread(target=self.camera_thread, daemon=True)
        camera_thread.start()
        
        # Setup and run matplotlib animation
        self.setup_plot()
        
        FPS = 24  # Giống mapping.py
        ani = FuncAnimation(self.fig, self.update_animation, init_func=self.init_animation,
                          interval=1000/FPS, blit=False, cache_frame_data=False)  # Changed blit=False for debugging
        
        plt.show()
        
        # Cleanup
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Close serial connection
        if self.serial_connection:
            self.serial_connection.close()
            print("Đã đóng kết nối serial")

def main():
    app = CameraMappingCombined()
    app.run()

if __name__ == "__main__":
    main()
