clear; clc; close all;

%% =========================================================
% 1. 座標系定義 (World, Table, DUT, Gyro)
%    運動學鏈: World -> Table -> DUT -> Gyro
%    Table 為 non-orthogonal 基底 (製造誤差)
% ==========================================================
err = deg2rad(5); % Table Z 軸誤差 (rad)

% World / Rotation Axis Frame (真值轉台，假設跟世界重合)
r_coor = eye(3);

% Table Frame: non-orthogonal 基底
% X 軸 [1,0,0] 與 Z 軸 [sin,0,cos] 不正交 (內積 = sin(err) ≠ 0)
t_coor = [1, 0,        sin(err);
          0, 1,        0;
          0, 0,        cos(err)];

% DUT Frame: 相對於 Table (正交安裝姿態)
R_dut = rotz(15) * roty(10) * rotx(5);

% Gyro Frame: 相對於 DUT (正交安裝姿態)
R_g = rotz(-45) * roty(30) * rotx(60);

% 真實轉軸 (世界下, 單位向量): Table 自身 Z 軸方向
z_axis = t_coor(:, 3);   % [sin(err); 0; cos(err)]

%% =========================================================
% 2. Rodrigues 正交旋轉: 轉台繞 z_axis 轉 θ
%    Z 軸固定不掃 cone，盤面繞斜軸轉動
% ==========================================================
theta = deg2rad(90); % 轉動角 (可改)
w     = 1;           % 角速度 (d/s)

% 繞 z_axis 轉 θ 的正交旋轉矩陣
R_turn = rodrigues(z_axis, theta);

%% =========================================================
% 3. DUT / Gyro 世界姿態 (正交, 不繼承 non-orthogonal 誤差)
% ==========================================================
DUT_world  = R_turn * R_dut;             % 正交
Gyro_world = R_turn * R_dut * R_g;       % 正交

%% =========================================================
% 4. Gyro 量測
%    角速度向量(世界下)沿真實轉軸: w_world = w * z_axis
%    gyro 局部量測 = Gyro_world' * w_world
%    因 R_turn'*z_axis = z_axis，量測與 θ 無關
% ==========================================================
w_world = w * z_axis;
w_gyro  = Gyro_world' * w_world;

fprintf('Table 基底 t_coor (non-orthogonal):\n'); disp(t_coor);
fprintf('DUT 相對 Table:\n');                 disp(R_dut);
fprintf('Gyro 相對 DUT:\n');                  disp(R_g);
fprintf('真實轉軸 z_axis (世界下):\n');       disp(z_axis);
fprintf('\nR_turn(θ=%g°):\n', rad2deg(theta)); disp(R_turn);
fprintf('R_turn 正交? %g\n', norm(R_turn'*R_turn - eye(3)));
fprintf('DUT_world (正交):\n');              disp(DUT_world);
fprintf('Gyro_world (正交):\n');             disp(Gyro_world);
fprintf('\n角速度向量 (world, 沿斜 Z 軸):\n');disp(w_world);
fprintf('Gyro 量到 (d/s):\n');               disp(w_gyro);

% Rodrigues: 繞單位軸 k 旋轉 a 度 (正交矩陣)
function R = rodrigues(k, a)
    c = cos(a); s = sin(a);
    K = [0, -k(3), k(2);
         k(3), 0, -k(1);
         -k(2), k(1), 0];
    R = c*eye(3) + (1-c)*(k*k') + s*K;
end