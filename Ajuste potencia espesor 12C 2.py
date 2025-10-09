import numpy as np
import matplotlib.pyplot as plt

def weighted_powerlaw_fit(t, y, s):
    t = np.asarray(t, float); y = np.asarray(y, float); s = np.asarray(s, float)
    if np.any((t <= 0) | (y <= 0)) or np.any(s <= 0):
        raise ValueError("t,y > 0 y sigma_y > 0 para el ajuste en log.")

    lnT, lnY = np.log(t), np.log(y)
    w = (y / s) ** 2  # ≈ 1/Var(ln y)

    X = np.vstack((lnT, np.ones_like(lnT))).T
    W = np.diag(w)
    XTW = X.T @ W
    cov_unscaled = np.linalg.inv(XTW @ X)
    beta = cov_unscaled @ (XTW @ lnY)  # [a, lnA]
    a, lnA = beta
    A = np.exp(lnA)

    yhat = A * (t ** a)
    chi2 = np.sum(((y - yhat) / s) ** 2)
    dof = len(t) - 2
    chi2_red = chi2 / dof

    # Incertidumbres (covarianza escalada por chi2_red)
    cov = cov_unscaled * chi2_red
    a_err = np.sqrt(cov[0, 0])
    A_err = A * np.sqrt(cov[1, 1])

    return dict(A=A, a=a, A_err=A_err, a_err=a_err,
                yhat=yhat, chi2=chi2, dof=dof, chi2_red=chi2_red)

# ================== DATOS ==================
ION = r"$^{12}\mathrm{C}$"
t = np.array([3, 6, 9, 12], float)  # espesor (mg/cm^2)

theta_1e   = np.array([0.847, 1.345, 1.816, 2.278], float)
sigma_1e   = np.array([0.011, 0.023, 0.009, 0.013], float)

theta_1_10 = np.array([1.379, 2.197, 2.927, 3.651], float)
sigma_1_10 = np.array([0.009, 0.019, 0.010, 0.010], float)

theta_1_100 = np.array([2.373, 3.660, 4.834, 5.996], float)
sigma_1_100 = np.array([0.009, 0.021, 0.009, 0.020], float)

series = [
    (r"$\theta_{1/e}$",   theta_1e,    sigma_1e),
    (r"$\theta_{1/10}$",  theta_1_10,  sigma_1_10),
    (r"$\theta_{1/100}$", theta_1_100, sigma_1_100),
]

# =============== FIGURA (ejes lineales) ===============
fig, ax = plt.subplots(figsize=(7.4, 4.8))

colors = ["C0", "C1", "C2"]
legend_handles = []
lines_text = []

r = np.ptp(t) if len(t) > 1 else 1.0
tt = np.linspace(t.min() - 0.05*r, t.max() + 0.05*r, 400)

for (label, y, s), color in zip(series, colors):
    res = weighted_powerlaw_fit(t, y, s)

    # puntos + barras (mismo color)
    ax.errorbar(t, y, yerr=s, fmt='o', capsize=3,
                color=color, ecolor=color, label="_nolegend_")

    # curva y = A t^a (mismo color)
    line, = ax.plot(tt, res["A"] * (tt ** res["a"]),
                    color=color, linewidth=2, label=label)
    legend_handles.append(line)

    # texto: A, a y chi2_red
    lines_text.append(
        f"{label}: A={res['A']:.3g}±{res['A_err']:.2g}, "
        f"a={res['a']:.3g}±{res['a_err']:.2g}, "
        f"$\\chi^2_\\mathrm{{red}}$={res['chi2_red']:.3f}"
    )

    # salida
    print(f"--- {label} ---")
    print(f"A = {res['A']:.8g} ± {res['A_err']:.8g}")
    print(f"a = {res['a']:.8g} ± {res['a_err']:.8g}")
    print(f"chi2 = {res['chi2']:.6f}   dof = {res['dof']}   chi2_red = {res['chi2_red']:.6f}\n")

ax.set_xlabel(r"Espesor (mg/cm$^2$)")
ax.set_ylabel(r"Ángulo (°)")
ax.set_title(r"$\theta_{1/e}$, $\theta_{1/10}$ y $\theta_{1/100}$ frente a espesor de la lámina")
ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

ax.text(0.02, 0.90, "\n".join(lines_text),
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"))

fig.tight_layout()
plt.savefig("powerlaw_thickness_12C.png", dpi=200)
plt.show()