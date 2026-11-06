"""
Heat Sink Physics Model + Parameter Sweep
Geometry: 90x116x27 mm, 60 fins @ 0.8 mm thick, 2.5 mm base
Die: 52.5 x 45 mm (from reference document)
Sweep: TDP (20 pts, 30-250 W), V_air (20 pts, 0.5-15 m/s), k_tim (10 pts, 1-12 W/m·K)
Area formula matches reference script exactly.
"""

import numpy as np
import pandas as pd

# --- Die dimensions (from reference document) ---
L_die = 0.0525      # die length [m]
W_die = 0.045       # die width [m]
A_die = L_die * W_die  # 0.002363 m²

# --- Heat sink geometry (fixed) ---
L = 90e-3           # heat sink length [m]
W = 116e-3          # heat sink width [m]
H = 27e-3           # heat sink total height [m]
Fin_Thickness = 0.8e-3
N_fins = 60
Base_Thickness = 2.5e-3
t_tim = 0.1e-3      # TIM thickness [m]

# --- Air properties at 25 °C (from reference script) ---
k_air = 0.0262      # [W/m·K]
nu_air = 1.57e-5    # kinematic viscosity [m²/s]
Pr_air = 0.71

# --- Fixed resistances ---
R_jc = 0.2          # junction-to-case [°C/W]
T_ambient = 25.0    # ambient temperature [°C]

# --- Pre-computed geometry (velocity-independent) ---
s_f = (W - (N_fins * Fin_Thickness)) / (N_fins - 1)   # fin spacing [m]
h_fin = H - Base_Thickness                              # fin height [m]
A_fin = N_fins * (2 * h_fin * L) + (s_f * L)           # fin faces + one base strip
A_base_convection = (L * W) - (Fin_Thickness * N_fins * L)  # exposed base floor
A_total = A_fin + A_base_convection                     # total convection area [m²]


def compute_thermal(TDP, V_air, k_tim):
    Re = (V_air * s_f) / nu_air
    Nu = 1.86 * ((Re * Pr_air * (2 * s_f) / L) ** (1/3)) if Re < 2300 \
         else 0.023 * (Re ** 0.8) * (Pr_air ** 0.3)
    h = (Nu * k_air) / (2 * s_f)
    R_conv = 1.0 / (h * A_total)
    R_tim  = t_tim / (k_tim * A_die)
    R_total = R_jc + R_tim + R_conv
    T_junction = T_ambient + TDP * R_total
    return Re, Nu, h, R_conv, R_tim, R_total, T_junction


# --- Parameter sweep (assessment ranges) ---
TDP_vals   = np.linspace(30, 250, 20)
V_vals     = np.linspace(0.5, 15, 20)
k_tim_vals = np.linspace(1.0, 12.0, 10)

rows = []
for TDP in TDP_vals:
    for V in V_vals:
        for k in k_tim_vals:
            Re, Nu, h, R_conv, R_tim, R_total, T_j = compute_thermal(TDP, V, k)
            rows.append({
                "TDP_W":       round(TDP, 4),
                "V_air_ms":    round(V, 4),
                "k_tim_WmK":   round(k, 4),
                "Re":          round(Re, 4),
                "Nu":          round(Nu, 6),
                "h_Wm2K":      round(h, 4),
                "R_conv":      round(R_conv, 6),
                "R_tim":       round(R_tim, 6),
                "R_total":     round(R_total, 6),
                "T_junction_C": round(T_j, 4),
            })

df = pd.DataFrame(rows)
df.to_csv("heat_sink_dataset.csv", index=False)
print(f"Dataset saved: {len(df)} rows")

# --- Sanity check vs Excel reference (TDP=150W, V=1m/s, k_tim=4.0) ---
# Note: Excel reference gives R_total=0.373043 C/W using fin-only area (A_fin=0.2646 m2).
# Provided Python script includes base convection (A_total=0.2708 m2), yielding 0.369205 C/W (1.03% diff).
_, _, _, _, _, R_def, T_def = compute_thermal(150, 1.0, 4.0)
assert abs(R_def - 0.373043) / 0.373043 < 0.02, f"Sanity check error exceeds 2%: {R_def}"
print(f"Sanity check  : R_total={R_def:.6f} °C/W  T_junction={T_def:.5f} °C")
print(f"Excel reference: R_total=0.373043 °C/W    T_junction=80.95652 °C")
print(f"Error          : R={abs(R_def-0.373043)/0.373043*100:.2f}%  T={abs(T_def-80.95652)/80.95652*100:.2f}%")
print(f"R_total range  : {df.R_total.min():.3f} – {df.R_total.max():.3f} °C/W")
print(f"T_junction range: {df.T_junction_C.min():.1f} – {df.T_junction_C.max():.1f} °C")

