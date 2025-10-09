import numpy as np
import matplotlib.pyplot as plt

def weighted_linear_fit(x, y, s):
    x = np.asarray(x, float); y = np.asarray(y, float); s = np.asarray(s, float)
    if np.any(s <= 0): raise ValueError("Todas las sigma_y deben ser > 0")
    w = 1.0/(s**2)
    X = np.vstack((x, np.ones_like(x))).T
    W = np.diag(w)
    cov_unscaled = np.linalg.inv(X.T @ W @ X)
    beta = cov_unscaled @ (X.T @ W @ y)
    m, b = beta
    yhat = m*x + b
    res = y - yhat
    chi2 = np.sum((res/s)**2)
    dof = len(x) - 2
    chi2_red = chi2/dof
    cov = cov_unscaled * chi2_red
    m_err = np.sqrt(cov[0,0]); b_err = np.sqrt(cov[1,1])
    return dict(m=m, b=b, m_err=m_err, b_err=b_err,
                yhat=yhat, chi2=chi2, dof=dof, chi2_red=chi2_red)

# === DATOS ===
E = np.array([60, 72, 84, 96], float) 
y1 = np.array([1.471, 1.068, 0.857, 0.719], float); s1 = np.array([0.015, 0.011, 0.011, 0.009], float) 
y2 = np.array([2.308, 1.675, 1.342, 1.127], float); s2 = np.array([0.012, 0.009, 0.009, 0.010], float) 
y3 = np.array([3.605, 2.608, 2.091, 1.754], float); s3 = np.array([0.013, 0.010, 0.010, 0.011], float)

x = 1.0 / E
series = [
    (r"$\theta_{1/e}$",   y1, s1),
    (r"$\theta_{1/10}$",  y2, s2),
    (r"$\theta_{1/100}$", y3, s3),
]

fig, ax = plt.subplots(figsize=(7,4.5))

# rango NumPy 2.x
r = np.ptp(x)
xx = np.linspace(x.min() - 0.02*r, x.max() + 0.02*r, 300)

# Colores
colors = ["C0", "C1", "C2"]
legend_handles = []
lines_text = []

for (label, y, s), color in zip(series, colors):
    res = weighted_linear_fit(x, y, s)

    # puntos + barras (mismo color)
    ax.errorbar(x, y, yerr=s, fmt='o', capsize=3,
                color=color, ecolor=color, label="_nolegend_")

    # línea (mismo color) y en la leyenda
    line, = ax.plot(xx, res["m"]*xx + res["b"],
                    color=color, linewidth=2, label=label)
    legend_handles.append(line)

    # SOLO chi^2_red
    lines_text.append(f"{label}: $\\chi^2_\\mathrm{{red}}$ = {res['chi2_red']:.3f}")

    # impresión por consola (opcional)
    print(f"--- {label} ---")
    print(f"chi2 = {res['chi2']:.6f}   dof = {res['dof']}   chi2_red = {res['chi2_red']:.6f}")

ax.set_xlabel(r"$1/E_0$ (MeV$^{-1}$)")
ax.set_ylabel("Ángulo (°)")
ax.set_title(r"$\theta_{1/e}$, $\theta_{1/10}$ y $\theta_{1/100}$ frente a $1/E_0$")

# Leyenda
ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

# Bloque con χ^2_red dentro del eje, bajo el título
ax.text(0.02, 0.90, "\n".join(lines_text),
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"))

fig.tight_layout()
plt.savefig("tres_series_lineal_1sobreE.png", dpi=200)
plt.show()