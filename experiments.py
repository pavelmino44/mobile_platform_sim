"""Серия экспериментов: влияние высоты центра масс на занос и опрокидывание."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import robot_sim as rs

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(IMG_DIR, exist_ok=True)


def img(name):
    return os.path.join(IMG_DIR, name)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#2b2a28", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "lines.linewidth": 2, "legend.frameon": False,
})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
RED, VIOLET, GRAY = "#e34948", "#6250d6", "#8a8984"
seq = LinearSegmentedColormap.from_list("b", ["#9cc3f0", "#2a78d6", "#0b2f5c"])

out = {}

# ===================== E1: разгон по кругу, 6 высот башни =====================
H1 = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
f, te = rs.ramp(0.5, 4.0, 0.05)
r1 = rs.simulate(H1, f, te)
hs, ev1 = r1["hist"], r1["ev"]
cols = [seq(x) for x in np.linspace(0, 1, len(H1))]

fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
panels = [("e", 100, "Радиальное отклонение (занос), см"),
          ("beta", 1, "Угол бокового скольжения β, °"),
          ("phi", 1, "Угол крена корпуса, °"),
          ("Nin", 100, "Нагрузка на внутренних колёсах, % от статической")]
for a, (key, k, lab) in zip(ax.flat, panels):
    for i, H in enumerate(H1):
        v = hs["v"][:, i]; y = hs[key][:, i] * k
        a.plot(v, y, color=cols[i], label=f"H={H:.1f} м (h={r1['h'][i]*100:.0f} см)")
        m = ~np.isnan(y)
        if m.any():
            j = np.where(m)[0][-1]
            if not np.isnan(ev1["v_tip"][i]):
                a.plot(v[j], y[j], "x", color=RED, ms=9, mew=2.5, zorder=5)
            elif not np.isnan(ev1["v_lost"][i]):
                a.plot(v[j], y[j], "o", color=ORANGE, ms=7, zorder=5)
    a.set_title(lab, loc="left", fontsize=10)
ax[0, 0].axhline(rs.E_SKID * 100, color=GRAY, ls="--", lw=1)
ax[0, 0].text(0.55, rs.E_SKID * 100 + 1.5, "порог заноса 5 см", color=GRAY, fontsize=8)
ax[0, 0].set_ylim(-5, 65)
ax[1, 0].set_ylim(-1, 35)
for a in ax[1]:
    a.set_xlabel("Скорость, м/с")
ax[0, 0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.9))
fig.suptitle("Эксперимент 1. Плавный разгон (0,05 м/с²) по кругу R = 1 м\n"
             "× — опрокидывание, ● — робот ушёл с траектории (отклонение > 60 см)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(img("fig1_ramp_curves.png")); plt.close(fig)

# траектории
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
th = np.linspace(0, 2 * np.pi, 300)
for a, i, title in ((ax[0], 0, "H = 0 м: занос (уход по спирали наружу)"),
                    (ax[1], 4, "H = 0,8 м: опрокидывание на траектории")):
    a.plot(np.cos(th), np.sin(th), color=GRAY, ls="--", lw=1, label="заданный круг")
    X, Y, v = hs["X"][:, i], hs["Y"][:, i], hs["v"][:, i]
    sc = a.scatter(X, Y, c=v, cmap=seq, s=2, vmin=0.5, vmax=3.5)
    m = ~np.isnan(X); j = np.where(m)[0][-1]
    a.plot(X[j], Y[j], "x" if i == 4 else "o", color=RED if i == 4 else ORANGE, ms=10, mew=2.5)
    a.set_aspect("equal"); a.set_title(title, loc="left", fontsize=10)
    a.set_xlim(-2.2, 2.2); a.set_ylim(-2.2, 2.2)
fig.colorbar(sc, ax=ax, label="скорость, м/с", shrink=0.8)
fig.savefig(img("fig5_trajectories.png"), bbox_inches="tight"); plt.close(fig)

# ===================== E2/E4: критические скорости от высоты, 3 значения трения =====================
H2 = np.round(np.arange(0, 1.0001, 0.05), 3)
MUS = [0.5, 0.8, 1.1]
HH = np.tile(H2, len(MUS)); MM = np.repeat(MUS, len(H2))
r2 = rs.simulate(HH, rs.ramp(0.5, 4.5, 0.05)[0], (4.5 - 0.5) / 0.05, mu0=MM)
ev2 = r2["ev"]; h2 = rs.com_height(H2)
sel = MM == 0.8

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
a = ax[0]
hc = np.linspace(h2.min(), h2.max(), 200)
a.plot(hc * 100, np.full_like(hc, rs.v_skid_analytic()), color=AQUA, ls=":", lw=1.5,
       label="теория: занос, √(μgR)")
a.plot(hc * 100, rs.v_roll_analytic(hc), color=RED, ls=":", lw=1.5,
       label="теория: опрокид. жёсткого корпуса, √(gRT/2h)")
Hc = (hc * rs.MASS - rs.M_BASE * rs.H_BASE - rs.M_MAST * rs.DECK - rs.M_TOP * rs.DECK) / (rs.M_MAST / 2 + rs.M_TOP)
a.plot(hc * 100, rs.v_lift_compliant(Hc), color=MAGENTA, ls="--", lw=1.5,
       label="теория: отрыв колёс с учётом крена шин")
a.plot(h2 * 100, ev2["v_skid"][sel], "o-", color=AQUA, ms=6, label="симуляция: занос (e > 5 см)")
a.plot(h2 * 100, ev2["v_lift"][sel], "s-", color=MAGENTA, ms=5, label="симуляция: отрыв внутренних колёс")
a.plot(h2 * 100, ev2["v_tip"][sel], "X-", color=RED, ms=7, label="симуляция: опрокидывание")
hx = rs.T / (2 * 0.8)
a.axvline(hx * 100, color=GRAY, lw=1)
a.text(hx * 100 + 0.4, 1.5, f"h* = T/(2μ) ≈ {hx*100:.1f} см", color="#52514e", fontsize=8)
a.set_xlabel("Высота центра масс h, см"); a.set_ylabel("Скорость, м/с")
a.set_ylim(1.4, 4.5)
a.set_title("Критические скорости (μ = 0,8, R = 1 м)", loc="left", fontsize=10)
a.legend(fontsize=7.5, loc="upper right")
sec = a.secondary_xaxis("top", functions=(
    lambda h: (h / 100 * rs.MASS - 0.36 - 0.05 - 0.2) / 2.25,
    lambda H: (rs.com_height(H)) * 100))
sec.set_xlabel("Высота башни H, м", fontsize=9)

a = ax[1]
mcols = {0.5: "#9cc3f0", 0.8: BLUE, 1.1: "#0b2f5c"}
for mu in MUS:
    s = MM == mu
    vs, vt = ev2["v_skid"][s], ev2["v_tip"][s]
    vcrit = np.fmin(vs, vt)
    is_tip = ~np.isnan(vt) & ~(vs < vt)
    a.plot(h2 * 100, vcrit, color=mcols[mu], lw=2, label=f"μ = {mu}")
    a.plot(h2[is_tip] * 100, vcrit[is_tip], "X", color=RED, ms=6)
    a.plot(h2[~is_tip] * 100, vcrit[~is_tip], "o", color=AQUA, ms=5)
    a.axvline(rs.T / (2 * mu) * 100, color=mcols[mu], lw=0.8, ls="--")
a.plot([], [], "o", color=AQUA, label="предел — занос")
a.plot([], [], "X", color=RED, label="предел — опрокидывание")
a.set_xlabel("Высота центра масс h, см"); a.set_ylabel("Предельная скорость, м/с")
a.set_title("Какой отказ наступает первым при разном трении", loc="left", fontsize=10)
a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(img("fig2_critical_speeds.png")); plt.close(fig)

# ускорения-пороги (безразмерно)
fig, a = plt.subplots(figsize=(6.5, 4.3))
a.plot(hc * 100, rs.T / (2 * hc), color=RED, ls=":", lw=1.5, label="SSF = T/(2h) (жёсткий корпус)")
a.axhline(0.8, color=AQUA, ls=":", lw=1.5, label="μ = 0,8")
a.plot(h2 * 100, ev2["ay_skid"][sel], "o-", color=AQUA, ms=6, label="симуляция: a_y/g при заносе")
a.plot(h2 * 100, ev2["ay_lift"][sel], "s-", color=MAGENTA, ms=5, label="симуляция: a_y/g при отрыве колёс")
a.set_ylim(0, 1.6); a.set_xlabel("Высота центра масс h, см"); a.set_ylabel("Боковое ускорение a_y / g")
a.set_title("Предельное боковое ускорение", loc="left", fontsize=10); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(img("fig3_lateral_accel.png")); plt.close(fig)

# ===================== E3: карта режимов, резкий выход на круг с постоянной скоростью =====================
Hg = np.round(np.arange(0, 1.0001, 0.05), 3)
Vg = np.round(np.arange(1.2, 3.41, 0.1), 2)
HHg, VVg = np.meshgrid(Hg, Vg, indexing="ij")
hf, vf = HHg.ravel(), VVg.ravel()
r3 = rs.simulate(hf, lambda t: (vf, 0.0), 6.0, v0=vf, record_every=10**9)
ev3 = r3["ev"]
code = np.zeros(hf.size, int)                                # 0 устойчиво
code[ev3["max_e"] > rs.E_SKID] = 1                           # занос
code[ev3["min_Nin"] <= 1e-6] = 2                             # отрыв колёс, но вернулся
code[~np.isnan(ev3["v_tip"])] = 3                            # опрокидывание
C = code.reshape(HHg.shape)

from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#e8f1fc", AQUA, YELLOW, RED])
fig, a = plt.subplots(figsize=(8, 5.2))
hg = rs.com_height(Hg)
a.pcolormesh(Vg, hg * 100, C, cmap=cmap, vmin=-0.5, vmax=3.5, shading="nearest",
             edgecolors="white", linewidth=0.6)
a.plot(ev2["v_tip"][sel], h2 * 100, "k-", lw=1.5, label="граница опрокидывания при плавном разгоне")
a.plot(ev2["v_skid"][sel], h2 * 100, "k--", lw=1.5, label="граница заноса при плавном разгоне")
for c, lab in zip(["#e8f1fc", AQUA, YELLOW, RED],
                  ["устойчивое движение", "занос (e > 5 см)", "отрыв колёс, робот устоял", "опрокидывание"]):
    a.plot([], [], "s", color=c, ms=10, label=lab, markeredgecolor="#8a8984", mew=0.5)
a.set_xlabel("Скорость, м/с"); a.set_ylabel("Высота центра масс h, см")
a.set_xlim(Vg[0] - 0.05, Vg[-1] + 0.05)
a.set_title("Эксперимент 3. Карта режимов: резкий выход на круг с заданной скоростью",
            loc="left", fontsize=10)
a.legend(fontsize=7.5, loc="upper right", frameon=True, facecolor="white", edgecolor="none")
fig.tight_layout(); fig.savefig(img("fig4_regime_map.png")); plt.close(fig)

# порог опрокидывания при резком входе по каждой высоте
v_tip_step = np.array([Vg[np.argmax(C[i] == 3)] if (C[i] == 3).any() else np.nan for i in range(len(Hg))])
v_lift_step = np.array([Vg[np.argmax(C[i] >= 2)] if (C[i] >= 2).any() else np.nan for i in range(len(Hg))])

# ===================== сохранение таблицы =====================
rows = []
for i, H in enumerate(H2):
    rows.append({"H_m": float(H), "h_cm": round(float(h2[i]) * 100, 1),
                 "SSF": round(rs.T / (2 * h2[i]), 2),
                 "v_skid": None if np.isnan(ev2["v_skid"][sel][i]) else round(float(ev2["v_skid"][sel][i]), 2),
                 "v_lift_ramp": None if np.isnan(ev2["v_lift"][sel][i]) else round(float(ev2["v_lift"][sel][i]), 2),
                 "v_tip_ramp": None if np.isnan(ev2["v_tip"][sel][i]) else round(float(ev2["v_tip"][sel][i]), 2),
                 "v_lift_step": None if np.isnan(v_lift_step[i]) else float(v_lift_step[i]),
                 "v_tip_step": None if np.isnan(v_tip_step[i]) else float(v_tip_step[i]),
                 "ay_skid": None if np.isnan(ev2["ay_skid"][sel][i]) else round(float(ev2["ay_skid"][sel][i]), 3),
                 "ay_lift": None if np.isnan(ev2["ay_lift"][sel][i]) else round(float(ev2["ay_lift"][sel][i]), 3)})
import csv
with open(os.path.join(os.path.dirname(IMG_DIR), "results_table.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
for mu in MUS:
    s = MM == mu
    print("mu", mu, "tip from H=", H2[~np.isnan(ev2["v_tip"][s])][:1], "skid speeds", np.round(ev2["v_skid"][s][:3], 2))
print("E1 events", {k: np.round(v, 2).tolist() for k, v in ev1.items() if k.startswith("v_")})
