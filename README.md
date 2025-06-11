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

# Build the package
After updating `build` with `python -m pip install --upgrade build` run
```bash
python -m build
```
The built package will be found inside `dist/`.

## Install package locally
After building the file you can install it locally with
```bash
pip install .\dist\cryofit-[VERSION].tar.gz 
```