import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ================== CARGA DE DATOS ==================
data = np.loadtxt(
    r"C:\Users\alber\OneDrive\Escritorio\Física Nuclear\TFM\Bibliografía\Pico de bragg\Programas regresiones\oxigeno\Oxigeno400_bragg_estudio_21_plot.dat"
)
z_start, z_end, D, err_percent = data.T

# Centros y anchos de bin
z = 0.5 * (z_start + z_end)
dz_bin = (z_end - z_start)

# ================== INCERTIDUMBRE EN D ==================
# 1) estadística dada por el archivo
sigma_D_stat = D * (err_percent / 100.0)

# 2) componente por discretización en z (pendiente local * ancho / sqrt(12))
m = np.zeros_like(D)
# diferencias centrales para interiores
m[1:-1] = (D[2:] - D[:-2]) / (z[2:] - z[:-2])
# extremos: una sola diferencia
m[0]  = (D[1] - D[0]) / (z[1] - z[0])
m[-1] = (D[-1] - D[-2]) / (z[-1] - z[-2])

sigma_D_disc = np.abs(m) * dz_bin / np.sqrt(12.0)

# 3) incertidumbre total en D
sigma_D = np.sqrt(sigma_D_stat**2 + sigma_D_disc**2)

# ================== AJUSTE PARABÓLICO ALREDEDOR DEL PICO ==================
idx_peak = np.argmax(D)
idx_fit  = np.arange(max(0, idx_peak-2), min(len(D), idx_peak+3))
z_fit    = z[idx_fit]
D_fit    = D[idx_fit]
sigma_fit = sigma_D[idx_fit]

def parabola(zv, a, b, c):
    return a*zv**2 + b*zv + c

params, cov = curve_fit(parabola, z_fit, D_fit, sigma=sigma_fit, absolute_sigma=True)
a, b, c = params

# Pico y dosis en el pico con COVARIANZAS
z_pico = -b / (2*a)
J_zp = np.array([ b/(2*a**2), -1/(2*a), 0.0 ])   # d z_pico / d(a,b,c)
var_z_pico = J_zp @ cov @ J_zp
sigma_z_pico = np.sqrt(var_z_pico)

D_pico = c - b**2/(4*a)
J_Dp = np.array([ (b**2)/(4*a**2), -b/(2*a), 1.0 ])  # d D_pico / d(a,b,c)
var_D_pico = J_Dp @ cov @ J_Dp
sigma_D_pico = np.sqrt(var_D_pico)

print(f"Pico de Bragg en z = {z_pico:.3f} ± {sigma_z_pico:.3f} cm")
print(f"Dosis en el pico = {D_pico:.9g} ± {sigma_D_pico:.3g}")

# ================== CRUCES Rx (INTERPOLACIÓN 2 PUNTOS) ==================
def zrx_y_sigma_two_points(z1, z2, D1, D2, s1, s2, Dx, sDx):
    """
    Interpolación lineal entre (z1,D1) y (z2,D2) para hallar zrx con D(zrx)=Dx.
    Propaga incertidumbre desde D1, D2 y Dx (independientes).
    Devuelve zrx, sigma_zrx y ∂zrx/∂Dx (para covarianzas entre Rx).
    """
    dz = z2 - z1
    Q  = D2 - D1
    if Q == 0:
        return np.nan, np.nan, np.nan
    N  = Dx - D1
    zrx = z1 + dz * N / Q

    # Derivadas parciales respecto a (D1, D2, Dx)
    dD1 = dz * (Dx - D2) / Q**2
    dD2 = -dz * (Dx - D1) / Q**2
    dDx = dz / Q

    var = (dD1*s1)**2 + (dD2*s2)**2 + (dDx*sDx)**2
    return zrx, np.sqrt(var), dDx

def indices_cruce(z_side, D_side, threshold, post=True):
    """
    Devuelve (i1,i2) tal que el umbral queda entre D_side[i1] y D_side[i2].
    Para 'post': perfil cae; para 'pre': perfil sube.
    """
    if post:
        idx = np.where(D_side < threshold)[0]
        if len(idx) == 0:
            return None
        i2 = idx[0]
        i1 = i2 - 1
    else:
        idx = np.where(D_side < threshold)[0]
        if len(idx) == 0:
            return None
        i1 = idx[-1]
        i2 = i1 + 1
    if i1 < 0 or i2 >= len(D_side):
        return None
    return i1, i2

def calcular_Rx(z, D, sD, z_pico, D_pico, sDp, frac, lado='post', nombre='R'):
    Dx = frac * D_pico
    sDx = abs(frac) * sDp

    if lado == 'post':
        mask = z > z_pico
        etiqueta = f"{nombre}{int(frac*100)} (post-pico)"
        post = True
    else:
        mask = z < z_pico
        etiqueta = f"{nombre}{int(frac*100)} (pre-pico)"
        post = False

    z_buscar = z[mask]
    D_buscar = D[mask]
    s_buscar = sD[mask]

    pair = indices_cruce(z_buscar, D_buscar, Dx, post=post)
    if pair is None:
        print(f"No se puede calcular {etiqueta}.")
        return np.nan, np.nan, np.nan

    i1, i2 = pair
    z1, z2 = z_buscar[i1], z_buscar[i2]
    D1, D2 = D_buscar[i1], D_buscar[i2]
    s1, s2 = s_buscar[i1], s_buscar[i2]

    z_rx, s_z_rx, d_dDx = zrx_y_sigma_two_points(z1, z2, D1, D2, s1, s2, Dx, sDx)
    if np.isnan(z_rx):
        print(f"No se puede calcular {etiqueta}.")
        return np.nan, np.nan, np.nan

    print(f"{etiqueta}: z = {z_rx:.3f} ± {s_z_rx:.4f} cm")
    return z_rx, s_z_rx, d_dDx

# === R80, R20, R50 (post) y R50 (pre) ===
z_r80,  s_r80,  dDx80       = calcular_Rx(z, D, sigma_D, z_pico, D_pico, sigma_D_pico, 0.8,  lado='post', nombre='R')
z_r20,  s_r20,  dDx20       = calcular_Rx(z, D, sigma_D, z_pico, D_pico, sigma_D_pico, 0.2,  lado='post', nombre='R')
z_r50_post, s_r50_post, dDx50_post = calcular_Rx(z, D, sigma_D, z_pico, D_pico, sigma_D_pico, 0.5,  lado='post', nombre='R')
z_r50_pre,  s_r50_pre,  dDx50_pre  = calcular_Rx(z, D, sigma_D, z_pico, D_pico, sigma_D_pico, 0.5,  lado='pre',  nombre='R')

# ================== FWHM y FALL-OFF (con covarianzas por Dpico) ==================
if not (np.isnan(z_r50_pre) or np.isnan(z_r50_post)):
    fwhm = z_r50_post - z_r50_pre
    cov_pre_post = dDx50_pre * dDx50_post * (0.25 * var_D_pico)
    var_fwhm = s_r50_post**2 + s_r50_pre**2 - 2.0 * cov_pre_post
    sigma_fwhm = np.sqrt(max(var_fwhm, 0.0))
    print(f"FWHM (anchura a media altura): {fwhm:.3f} ± {sigma_fwhm:.4f} cm")
else:
    fwhm = np.nan
    sigma_fwhm = np.nan
    print("No se puede calcular FWHM (falta R50 pre o post).")

if not (np.isnan(z_r80) or np.isnan(z_r20)):
    falloff = z_r20 - z_r80
    cov_20_80 = dDx20 * dDx80 * (0.16 * var_D_pico)
    var_falloff = s_r20**2 + s_r80**2 - 2.0 * cov_20_80
    sigma_falloff = np.sqrt(max(var_falloff, 0.0))
    print(f"Caída distal (R20-R80): {falloff:.3f} ± {sigma_falloff:.4f} cm")
else:
    falloff = np.nan
    sigma_falloff = np.nan

# ================== GRÁFICA ==================
plt.errorbar(z, D, yerr=sigma_D, fmt='o', label='Datos con error', capsize=3)
z_plot = np.linspace(z_fit.min(), z_fit.max(), 200)
plt.plot(z_plot, parabola(z_plot, *params), '-', label='Ajuste parabólico')

plt.axvline(z_pico, color='r', linestyle='--', label=f'Pico: z={z_pico:.2f}±{sigma_z_pico:.2f}')
plt.axhline(0.8*D_pico, color='g', linestyle='--', label='80% del pico')
plt.axvline(z_r80, color='m', linestyle='--', label=f'R80: z={z_r80:.2f}±{s_r80:.3f}')
plt.axhline(0.2*D_pico, color='b', linestyle='--', label='20% del pico')
plt.axvline(z_r20, color='c', linestyle='--', label=f'R20: z={z_r20:.2f}±{s_r20:.3f}')
plt.axhline(0.5*D_pico, color='orange', linestyle='--', label='50% del pico')
plt.axvline(z_r50_post, color='purple', linestyle='--', label=f'R50 post: z={z_r50_post:.2f}±{s_r50_post:.3f}')
plt.axvline(z_r50_pre,  color='brown',  linestyle='--', label=f'R50 pre: z={z_r50_pre:.2f}±{s_r50_pre:.3f}')

if not np.isnan(falloff):
    xc = 0.5*(z_r80+z_r20)
    plt.text(xc, 0.5*D_pico, f'Fall-off: {falloff:.2f}±{sigma_falloff:.3f} cm',
             color='k', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6))

plt.xlabel('Profundidad z (cm)')
plt.ylabel('Dosis')
plt.legend()
plt.tight_layout()
plt.show()