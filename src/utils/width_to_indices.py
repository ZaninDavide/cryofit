def width_to_indices(width, center, datax):  
    # TODO: remove hypothesis of equally spaced points
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
