# `cryofit`


```py
import numpy as np
import cryofit as cr
from scipy.signal import find_peaks
import math

f0 = 7436392514.0
QL = 1000
theta21 = 0
theta11 = 0
A = 1
k1 = 1
k2 = 2

freqs = np.linspace(f0 - 500e6, f0 + 500e6, num=int(1e4))
S11 = cr.model_S11(freqs, f0, QL, k1, k2, A, theta11)
S21 = cr.model_S21(freqs, f0, QL, k1, k2, A, theta21)
S11 += np.random.normal(0,1e-3,len(freqs))
S21 += np.random.normal(0,1e-3,len(freqs))

df = (freqs[-1] - freqs[0]) / len(freqs)
peaks, _ = find_peaks(np.abs(S21), height=np.abs(S21)[0]*100, distance=500e6/df)
peaks = [np.argmax(np.abs(S21))]

for id in peaks:
    idx_min = id - math.floor(50e6/df)
    idx_max = id + math.floor(50e6/df)
    res = cr.fit_resonance(freqs[idx_min:idx_max], S11[idx_min:idx_max], S21[idx_min:idx_max], 
        plot_S11_real = False,
        plot_S11_imag = False,
        plot_S11_abs = False,
        plot_S11_phase = False,
        plot_S21_real = False,
        plot_S21_imag = False,
        plot_S21_abs = True,
        plot_S21_phase = False,
    )

res = cr.fit_resonance(freqs, S11, S21, plot_S21_abs = True)
print(f"Qi = {res["params"]["Q_i"]["value"]}")
```

# Build & Install

## Option 1
```bash
pip install .
```

## Option 2
After updating `build` with `python -m pip install --upgrade build` run
```bash
python -m build
```
The built package will be found inside `dist/`. After building the file you can install it locally with
```bash
pip install .\dist\cryofit-[VERSION].tar.gz 
```