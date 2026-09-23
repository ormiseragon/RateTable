import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Rate Table 標定模型 — DUT 相對非正交 Table", layout="wide")


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
    "真實轉軸 = Table 自身 Z 軸。轉台繞此斜軸正交旋轉，帶動 DUT/Gyro。"
    "此版 R_dut 沿 Table 自身非正交軸合成 (非 yaw/pitch/roll，角度為耦合語意)。"
)

with st.sidebar:
    st.header("Table (轉台) 相對於 World")
    err = st.slider("Table Z 軸誤差 (deg)", 0.0, 30.0, 5.0, 0.1)

    st.header("DUT 相對於 Table (沿非正交軸)")
    st.caption("dut_rz/ry/rx 為繞 Table 自身斜軸 (t_z/t_y/t_x) 的耦合角, 非 yaw/pitch/roll")
    dut_rz = st.slider("DUT 繞 t_z (deg)", -180, 180, 15)
    dut_ry = st.slider("DUT 繞 t_y (deg)", -180, 180, 10)
    dut_rx = st.slider("DUT 繞 t_x (deg)", -180, 180, 5)

    st.header("Gyro 相對於 DUT")
    g_rz = st.slider("Gyro yaw (deg)", -180, 180, -45)
    g_ry = st.slider("Gyro pitch (deg)", -180, 180, 30)
    g_rx = st.slider("Gyro roll (deg)", -180, 180, 60)

    st.header("驅動轉台 (三軸定位)")
    omega = st.slider("角速度 ω (d/s)", 0.0, 10.0, 1.0, 0.1)
    axis_choice = st.radio("轉台繞軸 (一次一軸)", ["Z", "X", "Y"], index=0)
    target_theta = st.slider("目標角度 θ (deg)", 0.0, 360.0, 0.0, 1.0)

err = np.deg2rad(err)
t_coor = np.array([[1, 0, np.sin(err)],
                   [0, 1, 0],
                   [0, 0, np.cos(err)]])
R_g = rotz(g_rz) @ roty(g_ry) @ rotx(g_rx)

# DUT 相對於 Table: 沿 Table 自身非正交軸 (t_x/t_y/t_z) 合成, rz·ry·rx (rx 最先施)
# 非正交軸 -> 角度為耦合語意, 非 yaw/pitch/roll
t_x, t_y, t_z = t_coor[:, 0], t_coor[:, 1], t_coor[:, 2]
R_dut = (rodrigues(t_z, dut_rz)
         @ rodrigues(t_y, dut_ry)
         @ rodrigues(t_x, dut_rx))

# 轉台軸索引: Z->2, X->0, Y->1 (對應 Table 自身斜軸)
axis_idx = {"Z": 2, "X": 0, "Y": 1}[axis_choice]
base_axis = t_coor[:, axis_idx]

# R_turn 累積 (session_state): 從 identity 開始, 按下累積才左乘
if "R_turn" not in st.session_state:
    st.session_state["R_turn"] = np.eye(3)
R_turn = st.session_state["R_turn"].copy()

# 當下轉軸 (世界下): 已被累積姿態帶到當下位置
axis_world = R_turn @ base_axis

# 累積此段旋轉 (繞當下軸轉 target_theta, 左乘)
accumulate = st.button("累積此段旋轉", type="primary")
if accumulate:
    st.session_state["R_turn"] = rodrigues(axis_world, target_theta) @ st.session_state["R_turn"]
    R_turn = st.session_state["R_turn"].copy()
    axis_world = R_turn @ base_axis

# DUT / Gyro 世界姿態
DUT_world = R_turn @ R_dut
Gyro_world = R_turn @ R_dut @ R_g

# 角速度向量 (世界下) 沿當下轉軸
w_world = axis_world.reshape(3, 1) * omega

# gyro 自身座標下量到的值
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

axis_len = 4.0
disc_r = 6.0
origins = {
    "Table": np.array([0., 0, 0]),
    "DUT": np.array([5., 10, 2]),
    "Gyro": np.array([15., 12, 8]),
}
axis_colors = {"X": "red", "Y": "green", "Z": "blue"}
axis_lbl = {"X": "X", "Y": "Y", "Z": "Z"}


def add_frame_traces(fig, name, R, o, show_legend=True):
    for i, lbl in enumerate(["X", "Y", "Z"]):
        d = R[:, i] * axis_len
        fig.add_trace(go.Scatter3d(
            x=[o[0], o[0] + d[0]], y=[o[1], o[1] + d[1]], z=[o[2], o[2] + d[2]],
            mode="lines", line=dict(color=axis_colors[lbl], width=3),
            name=f"{name}-{lbl}" if show_legend else None,
            showlegend=show_legend and i == 0,
            legendgroup=name,
        ))
        fig.add_trace(go.Scatter3d(
            x=[o[0] + d[0] * 1.15], y=[o[1] + d[1] * 1.15], z=[o[2] + d[2] * 1.15],
            mode="text", text=[f"{name}-{lbl}"], textfont=dict(size=9),
            showlegend=False, legendgroup=name,
        ))


def add_disc_trace(fig, R, o, show_legend=True):
    ex = R[:, 0]
    ey = R[:, 1]
    ang = np.linspace(0, 2 * np.pi, 40)
    px = (ex[0] * np.cos(ang) + ey[0] * np.sin(ang)) * disc_r
    py = (ex[1] * np.cos(ang) + ey[1] * np.sin(ang)) * disc_r
    pz = (ex[2] * np.cos(ang) + ey[2] * np.sin(ang)) * disc_r
    fig.add_trace(go.Scatter3d(
        x=px + o[0], y=py + o[1], z=pz + o[2],
        mode="lines", line=dict(color="darkorange", width=2),
        name="table disc", showlegend=show_legend,
    ))


def add_axis_trace(fig, k, show_legend=True):
    fig.add_trace(go.Scatter3d(
        x=[0, k[0] * 6], y=[0, k[1] * 6], z=[0, k[2] * 6],
        mode="lines", line=dict(color="black", width=2, dash="dash"),
        name="rotation axis", showlegend=show_legend,
    ))


def build_plot(theta_deg):
    R_turn_t = rodrigues(axis_world, theta_deg) @ R_turn
    DUT_w = R_turn_t @ R_dut
    Gyro_w = R_turn_t @ R_dut @ R_g
    T_w = R_turn_t @ t_coor
    fig = go.Figure()
    add_frame_traces(fig, "Table", T_w, origins["Table"], show_legend=False)
    add_frame_traces(fig, "DUT", DUT_w, origins["DUT"], show_legend=False)
    add_frame_traces(fig, "Gyro", Gyro_w, origins["Gyro"], show_legend=True)
    add_disc_trace(fig, T_w, origins["Table"], show_legend=False)
    add_axis_trace(fig, axis_world, show_legend=False)
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                   aspectmode="data"),
        title="Coordinate frames (Red=X, Green=Y, Blue=Z) + table disc",
        showlegend=True,
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=1.0, y=1.15, xanchor="right",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=40, redraw=True),
                                      transition=dict(duration=0),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=True),
                                        mode="immediate")]),
            ],
        )],
    )
    fig.update_scenes(xaxis=dict(range=[-8, 20]), yaxis=dict(range=[-8, 20]),
                      zaxis=dict(range=[-8, 20]))
    return fig


n_frames = 60
if target_theta > 0:
    thetas = np.linspace(0, target_theta, n_frames, endpoint=True)
else:
    thetas = [0.0]
fig_anim = build_plot(thetas[0])
all_frames = []
for th in thetas:
    R_turn_t = rodrigues(axis_world, float(th)) @ R_turn
    DUT_w = R_turn_t @ R_dut
    Gyro_w = R_turn_t @ R_dut @ R_g
    T_w = R_turn_t @ t_coor
    fr_traces = []
    for name, R, o in [("Table", T_w, origins["Table"]),
                       ("DUT", DUT_w, origins["DUT"]),
                       ("Gyro", Gyro_w, origins["Gyro"])]:
        for i, lbl in enumerate(["X", "Y", "Z"]):
            d = R[:, i] * axis_len
            fr_traces.append(go.Scatter3d(
                x=[o[0], o[0] + d[0]], y=[o[1], o[1] + d[1]], z=[o[2], o[2] + d[2]],
                mode="lines", line=dict(color=axis_colors[lbl], width=3),
                showlegend=False))
            fr_traces.append(go.Scatter3d(
                x=[o[0] + d[0] * 1.15], y=[o[1] + d[1] * 1.15], z=[o[2] + d[2] * 1.15],
                mode="text", text=[f"{name}-{lbl}"], textfont=dict(size=9),
                showlegend=False))
    ex = T_w[:, 0]; ey = T_w[:, 1]
    ang = np.linspace(0, 2 * np.pi, 40)
    px = (ex[0] * np.cos(ang) + ey[0] * np.sin(ang)) * disc_r
    py = (ex[1] * np.cos(ang) + ey[1] * np.sin(ang)) * disc_r
    pz = (ex[2] * np.cos(ang) + ey[2] * np.sin(ang)) * disc_r
    fr_traces.append(go.Scatter3d(x=px, y=py, z=pz, mode="lines",
                                  line=dict(color="darkorange", width=2),
                                  showlegend=False))
    fr_traces.append(go.Scatter3d(x=[0, axis_world[0] * 6], y=[0, axis_world[1] * 6],
                                  z=[0, axis_world[2] * 6], mode="lines",
                                  line=dict(color="black", width=2, dash="dash"),
                                  showlegend=False))
    all_frames.append(go.Frame(data=fr_traces, name=f"{th:.0f}"))

fig_anim.update(frames=all_frames)

fig_anim.update_layout(
    sliders=[dict(steps=[dict(method="animate",
                              args=[[f.name], dict(mode="immediate",
                                                   frame=dict(duration=0))],
                              label=f.name) for f in fig_anim.frames],
                  transition=dict(duration=0))],
)
st.plotly_chart(fig_anim, use_container_width=True)

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
    f"- 目前累積姿態 R_turn 已旋轉 (累積的 Table 世界姿態)。"
)
st.write(
    "- 此版 R_dut 沿 Table 自身非正交軸 (t_x/t_y/t_z) 合成"
    " (rodrigues(t_z,rz)·rodrigues(t_y,ry)·rodrigues(t_x,rx)), "
    "故 dut_rz/ry/rx 為耦合角, 非 yaw/pitch/roll。"
)
st.write(
    "- 三軸轉台: radio 選一次繞一軸 (Z/X/Y)，動畫從 0 掃到目標 θ。"
    " 按下『累積此段旋轉』後，把繞『當下 Table 軸』轉 θ 左乘疊加到 R_turn:"
    " R_turn ← rodrigues(axis_world, θ)·R_turn，其中 axis_world = R_turn·t_coor[:,i]。"
)
st.write(
    f"- 當下轉軸 (世界下): axis_world = [{axis_world[0]:.4f}, {axis_world[1]:.4f},"
    f" {axis_world[2]:.4f}] (隨累積姿態漂移, 非固定)。"
)
st.write(
    "- gyro 量測: w_gyro = Gyro_worldᵀ·(ω·axis_world)；|w_gyro| = ω，"
    " 三分量隨姿態漂移而改變。"
)
st.caption(
    "non-orthogonal 誤差反映在 Table 自身軸，R_dut 沿斜軸合成, "
    "且轉台繞當下 Table 軸累積, 進而影響 gyro 量測的三軸分量。"
)