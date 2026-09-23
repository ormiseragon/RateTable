# 設計規格：轉台 Non-Orthogonal 誤差模型修正

**日期**：2026-09-21
**狀態**：已實作
**專案**：RateTable

## 1. 背景與動機

原 `sim_orth_dut.py`（舊名 `app.py`）模型將 non-orthogonal 基底 `t_coor = [1,0,sin(err); 0,1,0; 0,0,cos(err)]` 直接作為旋轉矩陣乘入 DUT/Gyro 姿態（`DUT_world = t_coor @ R_dut`），導致 DUT/Gyro 世界姿態被污染成 non-orthogonal。物理上 DUT（衛星）與 Gyro 是正交剛體，不應繼承轉台機構的製造誤差。同時，`RateTable.m` 註解「轉台 z 軸會繞出 cone」在「繞真實斜 Z 軸轉動」的模型下是錯誤的——Z 軸固定，不掃 cone。

## 2. 統一物理模型

### 2.1 Table 誤差基底
```
t_coor = [1,  0,       sin(err);
          0,  1,       0;
          0,  0,       cos(err)]
```
- X 軸 `[1,0,0]` 與 Z 軸 `[sin,0,cos]` 內積 = sin(err) ≠ 0 → non-orthogonal，僅描述轉台機構製造誤差。

### 2.2 真實轉軸（世界下，單位向量）
```
z_axis = t_coor[:, 2] = [sin(err), 0, cos(err)]
```
（長度 = 1，作為 Rodrigues 轉軸。）

### 2.3 轉台轉動（正交 Rodrigues）
轉台繞 `z_axis` 旋轉 θ，用 Rodrigues 公式得正交旋轉矩陣：
```
R_turn(θ) = cosθ·I + (1−cosθ)·k·kᵀ + sinθ·[k]ₓ ,   k = z_axis
```
性質（已驗證，θ ∈ {0,90,180,270}）：`R_turnᵀR_turn = I`、`det = 1`、`R_turn(θ)·k = k`（轉軸固定，不掃 cone）。

### 2.4 DUT / Gyro 世界姿態（正交，不繼承誤差）
```
DUT_world(θ)   = R_turn(θ) @ R_dut
Gyro_world(θ)  = R_turn(θ) @ R_dut @ R_g
```
全程正交。

### 2.5 Gyro 量測
```
w_gyro = Gyro_world(θ)ᵀ @ (ω · z_axis)
       = ω · R_gᵀ @ R_dutᵀ @ (R_turn(θ)ᵀ @ z_axis)
       = ω · R_gᵀ @ R_dutᵀ @ z_axis          (因 R_turnᵀ·k = k，θ 消去)
```
- 量測與 θ 無關（剛性帶動物理結果），已實證 θ ∈ {0,90,180,270} 結果相同。
- 反映 non-orthogonal 誤差（z_axis 含誤差）。
- 無「完整式 vs 簡化式」假差別（不再有 `t_coorᵀt_coor` 污染）。

## 3. `RateTable.m` 修改
1. `t_coor` 改用 non-orthogonal 基底 `[1,0,sin(err); 0,1,0; 0,0,cos(err)]`。
2. 移除「正交旋轉矩陣」「cone」錯誤描述，改為「轉台繞真實斜 Z 軸轉動，Z 軸固定不掃 cone」。
3. 實作 `rodrigues()` 函式。
4. 重算 `DUT_world = R_turn * R_dut`、`Gyro_world = R_turn * R_dut * R_g`、`w_gyro = Gyro_world' * (w*z_axis)`。
5. 移除無意義的 `w_gyro_simple`（對照式）。

## 4. `sim_orth_dut.py`（舊名 `app.py`）修改
1. 新增 `rodrigues()`。
2. 側邊欄移除「轉動軸 X/Y/Z」radio，新增「轉動角度 θ」滑桿（0–360°，預設 0）。
3. 新增 `z_axis = t_coor[:,2]`、`R_turn = rodrigues(z_axis, θ)`。
4. 修正 DUT/Gyro 世界姿態為正交 `R_turn@R_dut`、`R_turn@R_dut@R_g`。
5. 修正 gyro 量測為 `Gyro_worldᵀ @ (ω·z_axis)`。
6. 移除「完整式 vs 簡化式」差異區塊，改為「說明」區塊。
7. 3D 圖：新增轉台盤面（xy 盤法向量 = 當前轉軸方向，隨 θ 繞斜軸擺動）與真實轉軸虛線。
8. 靜態姿態矩陣：顯示 θ 下各座標系世界姿態，DUT/Gyro 正交。

## 5. 驗證準則
- `R_turn` 對任意 θ：正交、det=1、轉軸不變。
- 測試案例（使用者提供）：A=`[1,0,0.5; 0,1,0; 0,0,0.866]`，B=identity，θ=90° →
  `B_new = [0.25,−0.866,0.433; 0.866,0,−0.5; 0.433,0.5,0.75]`（已驗證）。
- gyro 量測與 θ 無關（θ ∈ {0,90,180,270} 數值相同）。

## 6. 後續（未實作，供參考）
- 可視需要加入自動旋轉動畫，展示盤面軌跡。
- 可加入「繞世界 Z 軸 vs 繞斜 Z 軸」兩種模式切換，對照 cone 行為。