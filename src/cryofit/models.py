import math
import numpy as np
from utils import peak_width, width_to_indices

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

def estimate_parameters(freqs, S11, S21):
    id_minS21 = np.argmin(np.abs(S21))
    id_maxS21 = np.argmax(np.abs(S21))
    minS21 = np.abs(S21[id_minS21])
    maxS21 = np.abs(S21[id_maxS21])

    f0 = freqs[id_maxS21]

    width = 0
    counter = 0
    for i in range(0, len(S21)):
        factor = np.abs(S21[i]) / maxS21
        threshold = 0.5 # bigger threshold => less points # TODO threshold = 1/2 is arbitrary
        if factor < threshold*minS21/maxS21 + (1-threshold)*1.0:
            k = 0.5 * np.sqrt(1/factor/factor - 1) # factor = 1 / np.sqrt(1 + 4*k*k)
            if k < 2: # TODO k < 2 is arbitrary
                width += np.abs(freqs[i] - f0) / k
                counter += 1
    if counter == 0: raise Exception("Could not estimate width")
    width /= counter

    QL = f0 / width

    theta21 = np.angle(S21)[id_maxS21] # Arg[S21(w_0)]
    theta11_up_to_pi = np.angle(S11)[id_maxS21] # Arg[S11(w_0)] up to pi due to sign of B

    (id_phi_min, id_phi_max) = width_to_indices(10 * width, f0, freqs)
    id_phi_min = id_maxS21 - max(1, id_maxS21 - id_phi_min) # we move at least one datapoint to the left
    id_phi_max = id_maxS21 + max(1, id_phi_max - id_maxS21) # we move at least one datapoint to the right
    id_phi_min = max(min(id_phi_min, len(freqs)-1), 0)
    id_phi_max = max(min(id_phi_max, len(freqs)-1), 0)
    B = 0
    for n in range(id_phi_min, id_phi_max + 1):
        if n != id_maxS21: 
            phi = np.angle(S11[n] / np.exp(1j * theta11_up_to_pi))
            f_phi = freqs[n]
            d_phi = f_phi/f0 - f0/f_phi
            B += (1 - np.tan(phi)*d_phi*QL) / (1 + np.tan(phi)/d_phi/QL)
    B /= id_phi_max + 1 - id_phi_min - 1 # average B estimate
    
    theta11 = theta11_up_to_pi
    if B < 0: theta11 = np.angle(np.exp(1j * (theta11_up_to_pi - np.pi))) # remove pi and make sure you are in [-pi, pi]

    S21_resonance = np.max(np.abs(S21))
    S11_resonance = np.abs(S11)[id_maxS21]
    
    A = S11_resonance / np.abs(B)
    C = S21_resonance / A

    alpha = (1 - B)/(1 + B)
    k1 = 4 / (4*alpha - (C*(alpha + 1))**2)
    k2 = alpha*k1 - 1

    return {
        "f0": f0,
        "QL": QL,
        "Qi": QL * (1 + k1 + k2),
        "k1": k1,
        "k2": k2,
        "A": A,
        "B": B,
        "C": C,
        "theta11": theta11,
        "theta21": theta21,
        "width": width
    }
