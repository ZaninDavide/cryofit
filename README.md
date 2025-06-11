# `cryofit`

```py
import cryofit as cr

f = ...
S21 = ...
S11 = ...

res = cr.fit_resonance(f, S11, S21, 
    plot_S11_real = False,
    plot_S11_imag = False,
    plot_S21_real = True,
    plot_S21_imag = False,
)
```