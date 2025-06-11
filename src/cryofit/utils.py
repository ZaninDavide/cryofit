import math
import numpy as np

def width_to_indeces(width, center, datax):  
    deltaf = ( datax[-1] - datax[0] ) / len(datax)
    min_f = center - width / 2.0
    max_f = center + width / 2.0
    min_f_idx = int((min_f - datax[0]) / deltaf)
    max_f_idx = int((max_f - datax[0]) / deltaf)
    min_f_idx = min(len(datax) - 1, max(0, min_f_idx))
    max_f_idx = min(len(datax) - 1, max(0, max_f_idx))

    if min_f_idx == max_f_idx:
        raise ValueError(f"min_index = max_index = {min_f_idx}, no data in this range")
    if min_f_idx > max_f_idx:
        raise ValueError("min_index > max_index, something went wrong")

    return (min_f_idx, max_f_idx)

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