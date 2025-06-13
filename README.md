# `cryofit`

```py
import cryofit as cr

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