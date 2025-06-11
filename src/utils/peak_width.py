import numpy as np
import math

def peak_width(datax, datay):
    """
    peak_width(datax, datay)

    Estimates the width of the peak at $1/\\sqrt(2)$ of the total height (-3dB).
    
    The peak is expected to be at the maximum of `datay`.
    `datax` is expected to be sorted either in increasing or decreasing order.
    """
    half_height_value = np.min(datay) + (np.max(datay) - np.min(datay)) / math.sqrt(2)
    hits = []
    above = datay[0] > half_height_value
    for i in range(1, len(datay)):
        new_above = datay[i] > half_height_value
        if new_above != above: 
            hits.append((datax[i] + datax[i-1]) / 2)
            above = new_above

    return abs(hits[-1] - hits[0])