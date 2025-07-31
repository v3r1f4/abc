figure;
axis equal;
xlim([-100 100]);
ylim([-100 100]);
hold on;

a = 16; % Chiều dài cạnh robot
l = sqrt(2*a); % Chiều dài đường chéo

% Điểm xuất phát
x_rb = 70;
y_rb = 10;

theta = deg2rad(80); % Chuyển từ độ sang radian

% Điểm kết thúc
x1 = -40;
y1 = -70;

% Vẽ điểm đích
plot(x1, y1, 'ro', 'MarkerSize', 2);

% Tạo hình vuông tâm tại gốc
x_r = [-1 1 1 -1] * a / 2;
y_r = [-1 -1 1 1] * a / 2;

% Ma trận xoay
R = [cos(theta), -sin(theta);
     sin(theta),  cos(theta)];
r = R * [x_r; y_r];

x_R = r(1,:) + x_rb;
y_R = r(2,:) + y_rb;

% Vẽ hình vuông
square = patch(x_R, y_R, [0.6 0.6 0.6], 'LineWidth', 1.5);

% Vẽ hướng
direction_line = plot([x_rb x_rb+(a/2)*cos(theta)], [y_rb y_rb+(a/2)*sin(theta)], 'r-', 'LineWidth', 2);
center_point = plot(x_rb, y_rb, 'k.', 'MarkerSize', 2);

A = [sqrt(2)/2,  sqrt(2)/2, sqrt(2)/2, sqrt(2)/2;
     sqrt(2)/2, -sqrt(2)/2, sqrt(2)/2, -sqrt(2)/2;
     -1/l,      -1/l,       1/l,       1/l];

pathX = x_rb;
pathY = y_rb;
path = plot(pathX, pathY, 'b-', 'LineWidth', 2);

dt = 0.01;
v = 15;

Kp = 2.5;
Ki = 0.05;
Kd = 0.3;

error_prev = 0;
error_integral = 0;
distance_threshold = 3;

x_text = -90;
y_text = 90;
t3xt = text(x_text, y_text, '', 'FontSize', 12);

for i = 1:1e6
    distance = sqrt((x1-x_rb)^2 + (y1-y_rb)^2);
    
    if distance < distance_threshold
        break;
    end
    
    % Tính góc hướng đến đích
    beta = atan2(y1-y_rb, x1-x_rb);
    
    % Sai số
    error = wrapToPi(beta - theta);
    
    % Điều khiển PID
    error_integral = error_integral + error * dt;
    error_derivative = (error - error_prev) / dt;
    omega = Kp * error + Ki * error_integral + Kd * error_derivative;
   
    
    % Điều chỉnh tốc độ theo khoảng cách
    v_current = min(v, max(5, distance * 2));
    
    % Tính tốc độ theo trục x, y
    vx = v_current * cos(theta);
    vy = v_current * sin(theta);
    
    % Cập nhật vị trí và hướng
    x_rb = x_rb + vx * dt;
    y_rb = y_rb + vy * dt;
    theta = theta + omega * dt;
    
    % Chuẩn hóa góc theta
    theta = wrapToPi(theta);

    P = [vx; vy; omega];
    V = pinv(A) * P;

    txt = sprintf("v1: %.3f, v2: %.3f, v3: %.3f, v4: %.3f\n", V(1, 1), V(2, 1), V(3, 1), V(4, 1));
    set(t3xt, 'String', txt);

    % Cập nhật ma trận xoay
    R = [cos(theta), -sin(theta);
         sin(theta),  cos(theta)];
    r = R * [x_r; y_r];
    x_R = r(1,:) + x_rb;
    y_R = r(2,:) + y_rb;

    % Cập nhật đồ họa
    set(square, 'XData', x_R, 'YData', y_R);
    set(direction_line, 'XData', [x_rb x_rb+(a/2)*cos(theta)], ...
            'YData', [y_rb y_rb+(a/2)*sin(theta)]);
    set(center_point, 'XData', x_rb, 'YData', y_rb);

    % Cập nhật đường đi
    pathX(end+1) = x_rb;
    pathY(end+1) = y_rb;
    set(path, 'XData', pathX, 'YData', pathY);

    % if mod(i, record_every_n_frames) == 0
    %    frame = getframe(gcf);
    %    writeVideo(v_writer, frame);
    %    frame_count = frame_count + 1;
    % end
    
    error_prev = error;
    pause(dt/10);
end

close(v_writer);

function out = wrapToPi(in)
    out = atan2(sin(in), cos(in));
end