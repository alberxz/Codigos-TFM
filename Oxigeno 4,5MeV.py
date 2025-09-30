import numpy as np
from pathlib import Path

def load_angular_file(path):
    """
    Formatos admitidos:
      - 4 columnas: theta_start  theta_end  I  err_percent
      - 3 columnas: theta  I  err_percent
      - 2 columnas: theta  I
    Devuelve:
      theta (centros), dtheta (ancho de bin), I, err_percent (o None)
    """
    M = np.loadtxt(path)
    if M.ndim != 2 or M.shape[1] not in (2, 3, 4):
        raise ValueError("Formato no reconocido (usa 2, 3 o 4 columnas).")

    if M.shape[1] == 4:
        th_start, th_end, I, errp = M.T
        theta = 0.5 * (th_start + th_end)
        dtheta = th_end - th_start
        return theta, dtheta, I, errp
    elif M.shape[1] == 3:
        theta, I, errp = M.T
        dtheta = np.empty_like(theta)
        dtheta[1:-1] = 0.5 * (theta[2:] - theta[:-2])
        dtheta[0] = theta[1] - theta[0]
        dtheta[-1] = theta[-1] - theta[-2]
        return theta, dtheta, I, errp
    else: 
        theta, I = M.T
        dtheta = np.empty_like(theta)
        dtheta[1:-1] = 0.5 * (theta[2:] - theta[:-2])
        dtheta[0] = theta[1] - theta[0]
        dtheta[-1] = theta[-1] - theta[-2]
        return theta, dtheta, I, None

#=============== Incertidumbre por bin: Monte Carlo + discretización =================
def sigma_I_total(theta, I, dtheta, err_percent=None, usar_discretizacion=True):
   
    # componente estadística (MC)
    if err_percent is None:
        sigma_stat = None
    else:
        sigma_stat = I * (np.asarray(err_percent) / 100.0)

    # componente por discretización en z
    m = np.zeros_like(I)
    m[1:-1] = (I[2:] - I[:-2]) / (theta[2:] - theta[:-2])
    m[0]    = (I[1]  - I[0])   / (theta[1]  - theta[0])
    m[-1]   = (I[-1] - I[-2])  / (theta[-1] - theta[-2])

    sigma_disc = np.abs(m) * dtheta / np.sqrt(12.0) if usar_discretizacion else 0.0

    if sigma_stat is None:
        return np.asarray(sigma_disc), False   # False = sin sigma MC absoluta
    else:
        sigma_tot = np.sqrt(sigma_stat**2 + np.asarray(sigma_disc)**2)
        return sigma_tot, True


# ================ Interpolación lineal a dos puntos con propagación ================
def cruce_dos_puntos(x1, x2, y1, y2, s1, s2, yf, syf):
    dx = x2 - x1
    Q  = y2 - y1
    if Q == 0:
        return np.nan, np.nan, np.nan
    xf = x1 + dx * (yf - y1) / Q

    # derivadas parciales
    d_y1 = dx * (yf - y2) / Q**2
    d_y2 = -dx * (yf - y1) / Q**2
    d_yf = dx / Q

    var = (d_y1 * s1)**2 + (d_y2 * s2)**2 + (d_yf * syf)**2
    return xf, np.sqrt(max(var, 0.0)), d_yf

# ================= Núcleo por INTERPOLACIÓN ==================
def angular_levels(path, niveles=(np.e**-1, 0.5, 0.1, 0.01),
                   lado='derecha', usar_discretizacion=True):
    theta, dtheta, I, errp = load_angular_file(path)
    sigmaI, have_mc_sigma = sigma_I_total(theta, I, dtheta, errp, usar_discretizacion)

    idx_peak = int(np.argmax(I))
    Ipk  = I[idx_peak]
    sIpk = sigmaI[idx_peak] if have_mc_sigma else 0.0
    thpk = theta[idx_peak]
    sThpk = dtheta[idx_peak] / np.sqrt(12.0)  # posición desconocida dentro del bin

    if lado.lower().startswith('d'):
        mask = theta >= thpk
    else:
        mask = theta <= thpk

    th_side = theta[mask]
    I_side  = I[mask]
    s_side  = sigmaI[mask] if have_mc_sigma else np.zeros_like(I_side)

    order = np.argsort(th_side)
    th_side, I_side, s_side = th_side[order], I_side[order], s_side[order]
    I_mono = np.maximum.accumulate(I_side[::-1])[::-1]

    resultados = {
        "theta_pico": float(thpk),
        "sigma_theta_pico": float(sThpk),
        "I_pico": float(Ipk),
        "sigma_I_pico": float(sIpk),
        "niveles": {}  # f -> (theta_f, sigma_theta_f)
    }

    for f in niveles:
        yf  = f * Ipk
        syf = abs(f) * sIpk

        # localiza el primer índice donde I cae por debajo del umbral yf
        below = np.where(I_mono < yf)[0]
        if len(below) == 0 or below[0] == 0:
            resultados["niveles"][float(f)] = (np.nan, np.nan)
            continue
        j2 = below[0]
        j1 = j2 - 1

        x1, x2 = th_side[j1], th_side[j2]
        y1, y2 = I_side[j1], I_side[j2]
        s1, s2 = s_side[j1], s_side[j2]

        thf, sthf, _ = cruce_dos_puntos(x1, x2, y1, y2, s1, s2, yf, syf)
        resultados["niveles"][float(f)] = (float(thf), float(sthf))

    return resultados

# ========================== Archivo =================================
if __name__ == "__main__":
    path = Path(r"C:\Users\alber\OneDrive\Escritorio\Física Nuclear\TFM\Bibliografía\Láminas delgadas\Estudio dispersión angular\Estudio Oxigeno\Distribución Ang_Oxigeno_Al_4,5MeV_21_tab.lis")

    print(f"CWD actual: {Path.cwd()}")
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo:\n  {path}")

    res = angular_levels(str(path),
                         niveles=(np.e**-1, 0.5, 0.1, 0.01),
                         lado='derecha',
                         usar_discretizacion=True)

    # ============================================================
    # En GRADOS
    # ============================================================
    DEG = 180.0/np.pi
    print(f"θ_pico = {res['theta_pico']*DEG:.6g}° ± {res['sigma_theta_pico']*DEG:.3g}°")
    print(f"I_pico = {res['I_pico']:.6g} ± {res['sigma_I_pico']:.3g}")
    for f in (np.e**-1, 0.5, 0.1, 0.01):
        th, sth = res['niveles'][float(f)]
        if np.isfinite(th):
            print(f"θ a {f:.5g}·I_pico : {th*DEG:.6g}° ± {sth*DEG:.3g}°")
        else:

            print(f"θ a {f:.5g}·I_pico : no evaluable")
