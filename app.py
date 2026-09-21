import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Rate Table 標定模型", layout="wide")


def rotx(a):
    a = np.deg2rad(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(a):
    a = np.deg2rad(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(a):
    a = np.deg2rad(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def rodrigues(k, theta_deg):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return c * np.eye(3) + (1 - c) * np.outer(k, k) + s * K


st.title("轉台 (Rate Table) 陀螺儀標定模型")
st.caption(
    "運動學鏈: World → Table → DUT → Gyro。Table 為 non-orthogonal 基底 (製造誤差)，"
    "真實轉軸 = Table 自身 Z 軸。轉台繞此斜軸正交旋轉，帶動 DUT/Gyro (保持正交)。"
)

with st.sidebar:
    st.header("Table (轉台) 相對於 World")
    err = st.slider("Table Z 軸誤差 (deg)", 0.0, 10.0, 5.0, 0.1)

    st.header("DUT 相對於 Table")
    dut_rz = st.slider("DUT yaw (deg)", -180, 180, 15)
    dut_ry = st.slider("DUT pitch (deg)", -180, 180, 10)
    dut_rx = st.slider("DUT roll (deg)", -180, 180, 5)

    st.header("Gyro 相對於 DUT")
    g_rz = st.slider("Gyro yaw (deg)", -180, 180, -45)
    g_ry = st.slider("Gyro pitch (deg)", -180, 180, 30)
    g_rx = st.slider("Gyro roll (deg)", -180, 180, 60)

    st.header("驅動轉台")
    theta = st.slider("轉動角度 θ (deg)", 0.0, 360.0, 0.0, 1.0)
    omega = st.slider("角速度 ω (d/s)", 0.0, 10.0, 1.0, 0.1)

err = np.deg2rad(err)
t_coor = np.array([[1, 0, np.sin(err)],
                   [0, 1, 0],
                   [0, 0, np.cos(err)]])
R_dut = rotz(dut_rz) @ roty(dut_ry) @ rotx(dut_rx)
R_g = rotz(g_rz) @ roty(g_ry) @ rotx(g_rx)

# 真實轉軸 (世界下, 單位向量): Table 自身 Z 軸方向
z_axis = t_coor[:, 2]

# 轉台繞 z_axis 轉 θ 的正交旋轉 (Rodrigues)
R_turn = rodrigues(z_axis, theta)

# DUT / Gyro 世界姿態 (正交, 不繼承 non-orthogonal 誤差)
DUT_world = R_turn @ R_dut
Gyro_world = R_turn @ R_dut @ R_g

# 角速度向量 (世界下) 沿真實轉軸
w_world = z_axis.reshape(3, 1) * omega

# gyro 自身座標下量到的值 (R_turn'@z_axis = z_axis, 故與 θ 無關)
w_gyro = Gyro_world.T @ w_world

# 各座標系下看到的角速度 (供對照)
w_in_table = t_coor.T @ w_world
w_in_dut = R_dut.T @ w_in_table
w_in_gyro = R_g.T @ w_in_dut

norm = np.linalg.norm(w_gyro)

col1, col2, col3 = st.columns(3)
col1.metric("wx", f"{w_gyro[0, 0]:.4f}", "d/s")
col2.metric("wy", f"{w_gyro[1, 0]:.4f}", "d/s")
col3.metric("wz", f"{w_gyro[2, 0]:.4f}", "d/s")
st.metric("大小 |w|", f"{norm:.4f}", "d/s")

st.header("Gyro 量到的角速度 (自身座標)")
fig, ax = plt.subplots(figsize=(6, 3))
labels = ["wx", "wy", "wz"]
colors = ["tab:red", "tab:green", "tab:blue"]
vals = [w_gyro[0, 0], w_gyro[1, 0], w_gyro[2, 0]]
xs = [0, 1, 2]
for x, v, c in zip(xs, vals, colors):
    ax.plot([-0.3, 2.3], [v, v], color=c, lw=2, alpha=0.9)
    ax.plot(x, v, "o", color=c, ms=7)
    ax.text(x, v, f" {v:.3f}", color=c, va="center", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels(labels)
ax.set_ylabel("d/s")
ax.set_ylim(-1.5, 1.5)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.grid(True, ls=":", alpha=0.4)
ax.set_title("Gyro angular rate (body frame)")
st.pyplot(fig)
plt.close(fig)

st.header("座標系 3D 視覺 (World / Table / DUT / Gyro)")
fig3d = plt.figure(figsize=(7, 6))
ax3 = fig3d.add_subplot(111, projection="3d")

frames = [
    ("World", np.eye(3), np.array([0., 0, 0])),
    ("Table", R_turn, np.array([0., 0, 0])),
    ("DUT", DUT_world, np.array([5., 10, 2])),
    ("Gyro", Gyro_world, np.array([15., 12, 8])),
]
axis_colors = {"X": "tab:red", "Y": "tab:green", "Z": "tab:blue"}
axis_len = 4.0
for name, R, o in frames:
    for i, (lbl, col) in enumerate(axis_colors.items()):
        d = R[:, i] * axis_len
        ax3.plot([o[0], o[0] + d[0]], [o[1], o[1] + d[1]], [o[2], o[2] + d[2]],
                 color=col, lw=2)
        ax3.text(o[0] + d[0] * 1.2, o[1] + d[1] * 1.2, o[2] + d[2] * 1.2,
                 f"{name}-{lbl}", fontsize=8)

# 轉台盤面: xy 平面 (法向量 = 當前轉軸方向)，隨 θ 繞斜軸擺動
disc_r = 6.0
ang = np.linspace(0, 2 * np.pi, 40)
ex = R_turn[:, 0]
ey = R_turn[:, 1]
px = ex[0] * np.cos(ang) + ey[0] * np.sin(ang)
py = ex[1] * np.cos(ang) + ey[1] * np.sin(ang)
pz = ex[2] * np.cos(ang) + ey[2] * np.sin(ang)
ax3.plot_surface(np.outer(px, [1, 1]) * disc_r,
                 np.outer(py, [1, 1]) * disc_r,
                 np.outer(pz, [1, 1]) * disc_r,
                 color="orange", alpha=0.25)
ax3.plot(disc_r * px, disc_r * py, disc_r * pz, color="darkorange", lw=1.5,
         label="table disc")

# 真實轉軸線
k = z_axis
ax3.plot([0, k[0] * 6], [0, k[1] * 6], [0, k[2] * 6],
         color="black", lw=2, ls="--", label="rotation axis")

ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z")
ax3.set_title("Coordinate frames (Red=X, Green=Y, Blue=Z) + table disc")
ax3.legend(loc="upper left", fontsize=8)
st.pyplot(fig3d)
plt.close(fig3d)

st.header("Static pose matrices")
show_parent = st.checkbox("Show parent-relative poses instead", value=False)

if show_parent:
    frames = [
        ("Table / World", R_turn),
        ("DUT / Table", R_dut),
        ("Gyro / DUT", R_g),
    ]
else:
    frames = [
        ("World", np.eye(3)),
        ("Table / World", R_turn),
        ("DUT / World", DUT_world),
        ("Gyro / World", Gyro_world),
    ]

cols = st.columns(len(frames))
for col, (name, M) in zip(cols, frames):
    with col:
        st.caption(name)
        st.dataframe(pd.DataFrame(np.round(M, 4),
                                  index=["X", "Y", "Z"],
                                  columns=["X", "Y", "Z"]),
                     use_container_width=True)

st.header("各座標系下看到的角速度 (d/s)")
df = pd.DataFrame({
    "X": [w_world[0, 0], w_in_table[0, 0], w_in_dut[0, 0], w_in_gyro[0, 0]],
    "Y": [w_world[1, 0], w_in_table[1, 0], w_in_dut[1, 0], w_in_gyro[1, 0]],
    "Z": [w_world[2, 0], w_in_table[2, 0], w_in_dut[2, 0], w_in_gyro[2, 0]],
}, index=["World", "Table", "DUT", "Gyro"])
st.dataframe(df, use_container_width=True)

st.divider()
st.subheader("說明")
st.write(
    f"- 真實轉軸 (世界下): z_axis = [{z_axis[0]:.4f}, {z_axis[1]:.4f}, {z_axis[2]:.4f}]"
    f"  (Table 自身 Z 軸, non-orthogonal)"
)
st.write(
    "- 轉台繞 z_axis 做正交 Rodrigues 旋轉 R_turn(θ)，DUT/Gyro 世界姿態全程正交"
    " (不繼承製造誤差)。"
)
st.write(
    "- gyro 量測與轉動角度 θ 無關: w_gyro = Gyro_worldᵀ·(ω·z_axis)"
    " = ω·R_gᵀ·R_dutᵀ·z_axis (因 R_turnᵀ·z_axis = z_axis)。"
)
st.caption(
    "non-orthogonal 誤差反映在真實轉軸 z_axis 方向，進而影響 gyro 量測的三軸分量。"
)