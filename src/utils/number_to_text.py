import numpy as np
from math import isnan

def number_to_text(x, sx, unit = "1", significant_figures=None):
    """
    number_to_text(x, sx, significant_figures=None)

    Parameters:
      - x is a number
      - sx is its uncertainty
      - significant_figures (optional) is the number of significant digits 
    """

    text = ""
    try:
        if significant_figures is not None and significant_figures <= 0: raise Exception("significant_figures must be None or a positive integer") 

        og = 0
        ogs = 0
        if x != 0 and not(isnan(x)): og = int(np.floor(np.log10(np.abs(x)))) # order of magnitude of x
        if sx != 0 and not(isnan(sx)): ogs = int(np.floor(np.log10(np.abs(sx)))) # order of magnitude of sx

        sx = round(sx / (10 ** ogs)) * (10 ** ogs) if not(isnan(sx)) else float("nan")

        # automatically calculate significant figures
        if significant_figures is None: significant_figures = max(1, 1 + og - ogs)        

        significant_figures -= 1

        mantissa_x = f"{x / (10 ** og):.{significant_figures}f}"
        mantissa_sx = f"{sx / (10 ** og):.{significant_figures}f}"

        if og == 0:
            text = f"({mantissa_x}\\ ±\\ {mantissa_sx})"
        else:
            text = f"({mantissa_x}\\ ±\\ {mantissa_sx}) × 10^{{{og}}}"
    except:
        text = f"({x}\\ ±\\ {sx})"

    if unit != "1" and unit != None: text = text + unit

    return text