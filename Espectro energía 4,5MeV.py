import numpy as np
from pathlib import Path

def leer_usr1d(path):
    """
    USR-1D con columnas: E_min  E_max  Y  err%
    Devuelve:
      E   = centros de bin (GeV)
      dE  = anchos de bin (GeV)
      Y   = densidad por energía en el bin
      ERR = error relativo en fracción
    """
    M = np.loadtxt(path, comments="#", usecols=(0,1,2,3))
    E1, E2, Y, ERRpct = M.T
    E   = 0.5*(E1 + E2)
    dE  = E2 - E1
    ERR = ERRpct / 100.0
    return E, dE, Y, ERR

def energia_salida_media(path):
    """
    Energía de salida como MEDIA del espectro y su incertidumbre (en GeV)
    """
    E, dE, Y, ERR = leer_usr1d(path)

    # Contenido por bin (área del histograma en ese bin)
    C = Y * dE
    S = C.sum()
    if S == 0:
        raise ValueError("Contenido total nulo; no puede calcularse la media.")

    # Media ponderada por contenido
    Emu = np.sum(E * C) / S

    # Propagación de incertidumbre desde err% (bins independientes)
    sC   = ERR * C
    varE = np.sum((E - Emu)**2 * sC**2) / (S**2)
    sEmu = np.sqrt(max(varE, 0.0))

    return Emu, sEmu

if __name__ == "__main__":
    path = Path(r"C:\Users\alber\OneDrive\Escritorio\Física Nuclear\TFM\Bibliografía\Láminas delgadas\Estudio dispersión angular\Estudio Oxigeno\Distribución Ang_Oxigeno_Al_4,5MeV_22_tab.lis")

    Emu_GeV, sEmu_GeV = energia_salida_media(path)

    GeV_to_MeV = 1000.0
    print(f"E_salida (media) = {Emu_GeV*GeV_to_MeV:.6g} MeV ± {sEmu_GeV*GeV_to_MeV:.3g} MeV")