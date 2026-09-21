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


st.title("轉台 (Rate Table) 陀螺儀標定模型")
st.caption(
    "各物體 (Table / DUT / Gyro) 有獨立的靜態世界姿態 (roll/pitch/yaw)。"
    "Table 帶有 non-orthogonal 製造誤差，轉軸 = Table 合成姿態的 Z 軸。"
    "轉台以角速度 ω 動態旋轉，問 gyro 量到多少角速度。"
)

with st.sidebar:
    st.header("Table (轉台) 世界姿態")
    t_roll = st.number_input("Table roll (deg)", -180.0, 180.0, 0.0, 0.1)
    t_pitch = st.number_input("Table pitch (deg)", -180.0, 180.0, 0.0, 0.1)
    t_yaw = st.number_input("Table yaw (deg)", -180.0, 180.0, 0.0, 0.1)
    err = st.number_input("Table Z 軸製造誤差 (deg)", 0.0, 30.0, 5.0, 0.1)

    st.header("DUT 世界姿態")
    d_roll = st.number_input("DUT roll (deg)", -180.0, 180.0, 5.0, 0.1)
    d_pitch = st.number_input("DUT pitch (deg)", -180.0, 180.0, 10.0, 0.1)
    d_yaw = st.number_input("DUT yaw (deg)", -180.0, 180.0, 15.0, 0.1)

    st.header("Gyro 世界姿態")
    g_roll = st.number_input("Gyro roll (deg)", -180.0, 180.0, 60.0, 0.1)
    g_pitch = st.number_input("Gyro pitch (deg)", -180.0, 180.0, 30.0, 0.1)
    g_yaw = st.number_input("Gyro yaw (deg)", -180.0, 180.0, -45.0, 0.1)

    st.header("動態旋轉")
    omega = st.slider("角速度 ω (d/s)", 0.0, 10.0, 1.0, 0.1)

# 各物體獨立靜態世界姿態 (正交)
Table_ini = rotz(t_yaw) @ roty(t_pitch) @ rotx(t_roll)
DUT_ini = rotz(d_yaw) @ roty(d_pitch) @ rotx(d_roll)
Gyro_ini = rotz(g_yaw) @ roty(g_pitch) @ rotx(g_roll)

# Table non-orthogonal 製造誤差基底
err = np.deg2rad(err)
t_coor = np.array([[1, 0, np.sin(err)],
                   [0, 1, 0],
                   [0, 0, np.cos(err)]])

# Table 合成姿態 + 轉軸 (動態轉動軸)
T_ini = Table_ini @ t_coor
z_axis = T_ini[:, 2]

# gyro 量測: 轉台繞 z_axis 以 ω 動態旋轉，投影到 gyro 世界姿態
w_world = z_axis.reshape(3, 1) * omega
w_gyro = Gyro_ini.T @ w_world

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
    ("Table", T_ini, np.array([0., 0, 0])),
    ("DUT", DUT_ini, np.array([5., 10, 2])),
    ("Gyro", Gyro_ini, np.array([15., 12, 8])),
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

# 轉台盤面: xy 平面 (法向量 = 轉軸 z_axis)
disc_r = 6.0
ex = T_ini[:, 0]
ey = T_ini[:, 1]
ang = np.linspace(0, 2 * np.pi, 40)
px = ex[0] * np.cos(ang) + ey[0] * np.sin(ang)
py = ex[1] * np.cos(ang) + ey[1] * np.sin(ang)
pz = ex[2] * np.cos(ang) + ey[2] * np.sin(ang)
ax3.plot_surface(np.outer(px, [1, 1]) * disc_r,
                 np.outer(py, [1, 1]) * disc_r,
                 np.outer(pz, [1, 1]) * disc_r,
                 color="orange", alpha=0.25)
ax3.plot(disc_r * px, disc_r * py, disc_r * pz, color="darkorange", lw=1.5,
         label="table disc")

# 動態轉軸線 (z_axis)
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
frames = [
    ("World", np.eye(3)),
    ("Table (合成)", T_ini),
    ("DUT", DUT_ini),
    ("Gyro", Gyro_ini),
]
cols = st.columns(len(frames))
for col, (name, M) in zip(cols, frames):
    with col:
        st.caption(name)
        st.dataframe(pd.DataFrame(np.round(M, 4),
                                  index=["X", "Y", "Z"],
                                  columns=["X", "Y", "Z"]),
                     use_container_width=True)

st.divider()
st.subheader("說明")
st.write(
    f"- 動態轉動軸 (世界下): z_axis = [{z_axis[0]:.4f}, {z_axis[1]:.4f}, {z_axis[2]:.4f}]"
    f"  = Table 合成姿態 Z 軸 (含製造誤差)"
)
st.write(
    "- 姿態與旋轉分離: Table/DUT/Gyro 各有獨立靜態世界姿態 (roll/pitch/yaw)；"
    "轉台繞 z_axis 以 ω 動態旋轉，不改變靜態姿態。"
)
st.write(
    f"- gyro 量測: w_gyro = Gyro_iniᵀ·(ω·z_axis)"
    f" = [{w_gyro[0,0]:.4f}, {w_gyro[1,0]:.4f}, {w_gyro[2,0]:.4f}] d/s"
)
st.caption(
    "non-orthogonal 製造誤差反映在 z_axis 方向，進而影響 gyro 量測的三軸分量。"
)