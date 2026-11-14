"""
Симуляция 4-колёсной мобильной платформы с "башней" на круговой траектории.

Модель (векторизована по пакету прогонов — каждый элемент массива = отдельный робот):
  * плоское движение корпуса: vx, vy, r (продольная/поперечная скорость, скорость рыскания), X, Y, psi;
  * крен корпуса phi вокруг оси на уровне пола (податливость шин/подвески k_phi, c_phi);
  * перераспределение нормальных нагрузок по 4 колёсам (поперечное — через момент крена,
    продольное — через m*ax*h/L), отрыв колёс при N <= 0;
  * шины: сила увода F = mu(N)*N*tanh(Cn*alpha/mu), эллипс трения с тяговой силой,
    чувствительность трения к нагрузке mu(N) = mu0*(1 - k_mu*(N/N0 - 1));
  * передние колёса управляемые (Аккерман), привод 4WD;
  * регулятор: прямая связь atan(L/R) + Stanley-подобная обратная связь по отклонению и курсу,
    П-регулятор скорости.
Критерии:
  * отрыв внутренних колёс (N_in = 0) — начало опрокидывания;
  * опрокидывание — угол крена превысил phi_tip = atan(T/(2h)) (ЦМ вышел за линию внешних колёс);
  * занос / потеря траектории — радиальное отклонение > E_SKID.
"""
import numpy as np

G = 9.81

# ---------------- фиксированные параметры платформы ----------------
L = 0.30        # колёсная база, м
T = 0.28        # колея, м
M_BASE = 6.0    # масса шасси, кг
H_BASE = 0.06   # высота ЦМ шасси, м
DECK = 0.10     # высота палубы (основания башни), м
M_MAST = 0.5    # масса мачты (равномерный стержень), кг
M_TOP = 2.0     # полезная нагрузка на вершине башни (сенсоры, манипулятор), кг
MASS = M_BASE + M_MAST + M_TOP

MU0 = 0.8       # коэффициент трения шина-пол
K_MU = 0.15     # чувствительность трения к нагрузке
CN = 8.0        # нормированная жёсткость увода, 1/рад (Calpha = CN*N)
K_PHI = 150.0   # жёсткость по крену (шины/подвеска), Н*м/рад
ZETA_PHI = 0.3  # относительное демпфирование крена

R_PATH = 1.0    # радиус круга, м
DELTA_MAX = np.radians(35)
E_SKID = 0.05   # порог "заноса" — радиальное отклонение, м
E_LOST = 0.60   # робот окончательно ушёл с траектории — прекращаем прогон


def com_height(H):
    """Высота центра масс в зависимости от высоты башни H."""
    H = np.asarray(H, float)
    return (M_BASE * H_BASE + M_MAST * (DECK + H / 2) + M_TOP * (DECK + H)) / MASS


def inertias(H):
    H = np.asarray(H, float)
    h = com_height(H)
    # крен вокруг оси на полу
    Ig = (M_BASE * (H_BASE**2 + (T**2 + 0.08**2) / 12)
          + M_MAST * ((DECK + H / 2)**2 + H**2 / 12)
          + M_TOP * ((DECK + H)**2 + 0.1**2 / 12))
    Iz = M_BASE * (L**2 + T**2) / 12 + M_TOP * 0.1**2 / 6 + 0.0 * H
    c_phi = 2 * ZETA_PHI * np.sqrt(K_PHI * Ig)
    return h, Ig, Iz, c_phi


def simulate(H, v_profile, t_end, mu0=MU0, dt=5e-4, v0=None, record_every=20):
    """
    H         : массив высот башни (пакет прогонов)
    v_profile : функция t -> (v_ref, a_ref) (массивы или скаляры)
    v0        : начальная скорость (по умолчанию v_ref(0))
    Возвращает dict с историей и событиями.
    """
    H = np.atleast_1d(np.asarray(H, float))
    n = H.size
    mu0 = np.broadcast_to(np.asarray(mu0, float), (n,)).copy()
    h, Ig, Iz, c_phi = inertias(H)
    phi_tip = np.arctan(T / (2 * h))
    N0 = MASS * G / 4

    vr0, _ = v_profile(0.0)
    v_init = np.broadcast_to(np.asarray(vr0 if v0 is None else v0, float), (n,)).copy()

    # старт на окружности в точке (R, 0), курс по касательной (против часовой)
    X = np.full(n, R_PATH); Y = np.zeros(n); psi = np.full(n, np.pi / 2)
    vx = v_init.copy(); vy = np.zeros(n); r = vx / R_PATH
    phi = np.zeros(n); dphi = np.zeros(n)
    ax_prev = np.zeros(n)

    xi = np.array([L / 2, L / 2, -L / 2, -L / 2])      # FL FR RL RR
    yi = np.array([T / 2, -T / 2, T / 2, -T / 2])
    steer = np.array([1, 1, 0, 0], float)

    alive = np.ones(n, bool)
    ev = {k: np.full(n, np.nan) for k in
          ("v_lift", "v_tip", "v_skid", "v_lost", "t_lift", "t_tip", "t_skid",
           "ay_lift", "ay_tip", "ay_skid")}
    ev["max_phi"] = np.zeros(n); ev["max_e"] = np.zeros(n); ev["min_Nin"] = np.ones(n)

    steps = int(round(t_end / dt))
    hist = {k: [] for k in ("t", "v", "e", "phi", "beta", "Nin", "ay", "X", "Y")}

    delta_ff = np.arctan(L / np.sqrt(R_PATH**2 - (L / 2)**2))
    for k in range(steps):
        t = k * dt
        v_ref, a_ref = v_profile(t)
        v_ref = np.broadcast_to(v_ref, (n,)); a_ref = np.broadcast_to(a_ref, (n,))

        # ---- регулятор ----
        rho = np.hypot(X, Y)
        e = rho - R_PATH
        th = np.arctan2(Y, X)
        course = psi + np.arctan2(vy, np.maximum(vx, 0.05))      # направление вектора скорости
        epsi = (course - (th + np.pi / 2) + np.pi) % (2 * np.pi) - np.pi
        vabs = np.maximum(np.hypot(vx, vy), 0.2)
        delta = np.clip(delta_ff - epsi + np.arctan2(2.0 * e, vabs), -DELTA_MAX, DELTA_MAX)
        Fx_cmd = MASS * (a_ref + 3.0 * (v_ref - vx))

        # ---- нормальные нагрузки ----
        M_spring = K_PHI * phi + c_phi * dphi
        M_max = MASS * G * T / 2
        M_g = np.clip(M_spring, -M_max, M_max)        # момент, передаваемый полом
        dN_lat = M_g / T                              # +: к правым (внешним при левом повороте)
        dN_lon = MASS * ax_prev * h / L / 2           # на колесо
        Nl = MASS * G / 2 - dN_lat
        Nr = MASS * G / 2 + dN_lat
        N = np.stack([Nl / 2 - dN_lon, Nr / 2 - dN_lon, Nl / 2 + dN_lon, Nr / 2 + dN_lon], 1)
        N = np.maximum(N, 0.0)

        # ---- шины ----
        # Аккерман: внутреннее (левое) колесо поворачивается сильнее
        tdl = np.tan(delta)
        dL = np.arctan(L * tdl / (L - T / 2 * tdl))
        dR = np.arctan(L * tdl / (L + T / 2 * tdl))
        z = np.zeros(n)
        d = np.stack([dL, dR, z, z], 1)
        cd, sd = np.cos(d), np.sin(d)
        vxi = vx[:, None] - r[:, None] * yi
        vyi = vy[:, None] + r[:, None] * xi
        vl = vxi * cd + vyi * sd
        vt = -vxi * sd + vyi * cd
        alpha = np.arctan2(vt, np.abs(vl) + 0.05)
        mu = mu0[:, None] * np.clip(1 - K_MU * (N / N0 - 1), 0.3, None)
        Fmax = mu * N
        Ft = -Fmax * np.tanh(CN * alpha / mu)
        Fl = np.broadcast_to((Fx_cmd / 4)[:, None], N.shape).copy()
        Ftot = np.hypot(Fl, Ft)
        scale = np.where(Ftot > Fmax, Fmax / np.maximum(Ftot, 1e-9), 1.0)
        Fl *= scale; Ft *= scale
        Fxb = Fl * cd - Ft * sd
        Fyb = Fl * sd + Ft * cd
        SFx = Fxb.sum(1); SFy = Fyb.sum(1)
        Mz = (xi * Fyb - yi * Fxb).sum(1)

        ax = SFx / MASS; ay = SFy / MASS
        # ---- интегрирование (полуявный Эйлер) ----
        vx_n = vx + dt * (ax + r * vy)
        vy_n = vy + dt * (ay - r * vx)
        r_n = r + dt * Mz / Iz
        ddphi = (MASS * ay * h * np.cos(phi) + MASS * G * h * np.sin(phi) - M_g) / Ig
        dphi_n = dphi + dt * ddphi
        phi_n = phi + dt * dphi_n
        psi_n = psi + dt * r_n
        X_n = X + dt * (vx_n * np.cos(psi_n) - vy_n * np.sin(psi_n))
        Y_n = Y + dt * (vx_n * np.sin(psi_n) + vy_n * np.cos(psi_n))

        a = alive
        vx[a], vy[a], r[a] = vx_n[a], vy_n[a], r_n[a]
        dphi[a], phi[a], psi[a] = dphi_n[a], phi_n[a], psi_n[a]
        X[a], Y[a] = X_n[a], Y_n[a]
        ax_prev = ax

        # ---- события ----
        v = np.hypot(vx, vy)
        Nin_frac = np.maximum(Nl, 0.0) / (MASS * G / 2)   # доля нагрузки на внутренних колёсах
        ev["min_Nin"] = np.where(a, np.minimum(ev["min_Nin"], Nin_frac), ev["min_Nin"])
        ev["max_phi"] = np.where(a, np.maximum(ev["max_phi"], phi), ev["max_phi"])
        ev["max_e"] = np.where(a, np.maximum(ev["max_e"], np.abs(e)), ev["max_e"])
        for name, cond in (("lift", Nin_frac <= 1e-6), ("tip", phi >= phi_tip),
                           ("skid", np.abs(e) > E_SKID)):
            new = a & cond & np.isnan(ev["v_" + name])
            ev["v_" + name][new] = v[new]
            ev["ay_" + name][new] = ay[new] / G
            ev["t_" + name][new] = t
        lost = a & (np.abs(e) > E_LOST)
        ev["v_lost"][lost] = v[lost]
        alive = alive & ~(phi >= phi_tip) & ~lost

        if k % record_every == 0:
            beta = np.degrees(np.arctan2(vy, np.maximum(vx, 1e-3)))
            for key, val in (("t", np.full(n, t)), ("v", np.where(a, v, np.nan)),
                             ("e", np.where(a, e, np.nan)),
                             ("phi", np.where(a, np.degrees(phi), np.nan)),
                             ("beta", np.where(a, beta, np.nan)),
                             ("Nin", np.where(a, Nin_frac, np.nan)),
                             ("ay", np.where(a, ay / G, np.nan)),
                             ("X", np.where(a, X, np.nan)), ("Y", np.where(a, Y, np.nan))):
                hist[key].append(val.copy())
        if not alive.any():
            break

    hist = {k: np.array(v) for k, v in hist.items()}
    return {"H": H, "h": h, "phi_tip": phi_tip, "mu0": mu0, "hist": hist, "ev": ev}


def ramp(v_start, v_end, accel):
    def f(t):
        v = min(v_start + accel * t, v_end)
        return v, (accel if v < v_end else 0.0)
    return f, (v_end - v_start) / accel


# ---------------- аналитические оценки (квазистатика) ----------------
def v_skid_analytic(mu=MU0, R=R_PATH):
    return np.sqrt(mu * G * R)


def v_roll_analytic(h, R=R_PATH):
    """Жёсткий корпус: a_y/g = T/(2h) (статический фактор устойчивости, SSF)."""
    return np.sqrt(G * R * T / (2 * h))


def v_lift_compliant(H, R=R_PATH):
    """С учётом крена на податливых шинах: m*ay*h = (k_phi - m g h)*phi, отрыв при k_phi*phi = mgT/2."""
    h = com_height(H)
    phi_lift = MASS * G * T / 2 / K_PHI
    ay = (MASS * G * T / 2 - MASS * G * h * np.sin(phi_lift)) / (MASS * h * np.cos(phi_lift))
    return np.sqrt(ay * R)
