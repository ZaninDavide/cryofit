import numpy as np
import math

def peak_width(datax, datay, factor = 1 / math.sqrt(2), minimum = None):
    """
    peak_width(datax, datay)

    Estimates the width of the peak at `factor` times the peak maximum height.
    
    The peak is expected to be at the maximum of `datay`.
    `datax` is expected to be sorted either in increasing or decreasing order.
    """
    minimum = minimum if minimum != None else np.min(datay)
    half_height_value = minimum + (np.max(datay) - np.min(datay)) * factor
    
    def hits(id1, id2):
        hits = np.array([])
        above = datay[id1] > half_height_value
        for i in range(id1+1, id2):
            new_above = datay[i] > half_height_value
            if new_above != above or datay[i] == half_height_value:
                a = np.abs(half_height_value - datay[i-1]) / (np.abs(datay[i]) - np.abs(datay[i-1]))
                hits = np.append(hits, a*datax[i] + (1-a)*datax[i-1])
                above = new_above
        return hits

    hits_before = hits(0, np.argmax(datay))
    hits_after = hits(np.argmax(datay), len(datay))

    if len(hits_before) == 0 or len(hits_after) == 0: return None

    return abs(np.mean(hits_after) - np.mean(hits_before))