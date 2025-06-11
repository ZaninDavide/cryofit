import math
import numpy as np
from utils import peak_width, width_to_indeces

# Characterization of superconducting resonant RF cavities for axion search with the QUAX experiment (Alessio Rettaroli Master Thesis)
# eq. (2.54)
def model_S11(f, f0, QL, k1, k2, A, theta11):
    delta = f/f0 - f0/f
    B = (k1-1-k2) / (1+k1+k2) # ex sqrtG
    return np.exp(1j * theta11) * A * (B + 1j*QL*delta) / (1 + 1j * QL * delta)

# eq. (2.57)
def model_S21(f, f0, QL, k1, k2, A, theta21):
    delta = f/f0 - f0/f
    C = 2 * math.sqrt(k1*k2) / (1+k1+k2) # ex sqrtI
    return np.exp(1j * theta21) * A * C / (1 + 1j*QL*delta)

def estimate_parameters(freqs, S21, S11):
    f0 = freqs[np.argmax(np.abs(S21))]
    width = peak_width(freqs, np.abs(S21)) # no - in front of datay

    QL = f0 / width # loaded quality factor

    theta21 = np.angle(S21)[np.argmax(np.abs(S21))] # Arg[S21(w_0)]
    theta11_up_to_pi = np.angle(S11)[np.argmax(np.abs(S21))] # Arg[S11(w_0)] up to pi due to sign of B

    (id_phi, _) = width_to_indeces(width / 3, f0, freqs)
    id_phi = np.argmax(np.abs(S21)) - max(1, np.argmax(np.abs(S21)) - id_phi) # we move at least one datapoint to the left
    phi = np.angle(S11[id_phi] / np.exp(1j * theta11_up_to_pi))
    f_phi = freqs[id_phi]
    d_phi = f_phi/f0 - f0/f_phi
    B = (1 - np.tan(phi)*d_phi*QL) / (1 + np.tan(phi)/d_phi/QL)

    theta11 = theta11_up_to_pi
    if B < 0: theta11 = np.angle(np.exp(1j * (theta11_up_to_pi - np.pi))) # remove pi and make sure you are in [-pi, pi]

    S21_resonance = np.max(np.abs(S21))
    S11_resonance = np.abs(S11)[np.argmax(np.abs(S21))]
    
    A = S11_resonance / np.abs(B)
    C = S21_resonance / A

    alpha = (1 - B)/(1 + B)
    k1 = 4 / (4*alpha - (C*(alpha + 1))**2)
    k2 = alpha*k1 - 1

    return {
        "f0": f0,
        "QL": QL,
        "k1": k1,
        "k2": k2,
        "A": A,
        "B": B,
        "C": C,
        "theta11": theta11,
        "theta21": theta21,
        "width": width
    }
