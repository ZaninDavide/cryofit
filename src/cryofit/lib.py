import numpy as np
from utils import width_to_indices
from cryofit.models import model_S21, model_S11, estimate_parameters
from fitter import Fitter, simultaneous_fit

def fit_resonance(
    freqs, S11, S21,
    plot_S11_real = False,
    plot_S11_imag = False,
    plot_S11_abs = False,
    plot_S11_phase = False,
    plot_S21_real = False,
    plot_S21_imag = False,
    plot_S21_abs = False,
    plot_S21_phase = False,
):
    guess = estimate_parameters(freqs, S21, S11)
    f0 = guess["f0"]
    QL = guess["QL"]
    k1 = guess["k1"]
    k2 = guess["k2"]
    A = guess["A"]
    width = guess["width"]
    theta11 = guess["theta11"]
    theta21 = guess["theta21"]

    # import pprint
    # pprint.pprint(guess)

    # ============================ REAL PARTS FITTERS ============================

    crop = width_to_indices(5 * width, f0, freqs)

    f21_real = Fitter()
    f21_real.datax = freqs[crop[0]:crop[1]] # Hz
    f21_real.datay = np.real(S21[crop[0]:crop[1]]) # 1
    f21_real.sigmay = f21_real.datay * 0 + 1e-3
    f21_real.model = lambda x, f0, QL, k1, k2, A, theta21: np.real(model_S21(x, f0, QL, k1, k2, A, theta21))

    f11_real = Fitter()
    f11_real.datax = freqs[crop[0]:crop[1]] # Hz
    f11_real.datay = np.real(S11[crop[0]:crop[1]]) # 1
    f11_real.sigmay = f11_real.datay * 0 + 1e-3
    f11_real.model = lambda x, f0, QL, k1, k2, A, theta11: np.real(model_S11(x, f0, QL, k1, k2, A, theta11))


    # ============================ PREPARING FOR FITTING ============================

    f21_real.scaley = "linear" # "linear" (default), "log", "dB"
    f21_real.unity = "1"
    f21_real.scalex = lambda x: x / 1e9 # "linear" (default), "log", "dB"
    f21_real.unitx = "GHz"
    f21_real.title = "Fit of $\\text{Re}(S_{21})$"
    f21_real.labelx = "Frequency"
    f21_real.labely = "$\\text{Re}(S_{21})$"
    f21_real.show_initial_model = True
    f21_real.show_plot = True
    f21_real.show_pvalue = False
    f21_real.show_model = True
    f21_real.figure_size = (30, 24)

    f11_real.scaley = "linear" # "linear" (default), "log", "dB"
    f11_real.unity = "1"
    f11_real.scalex = lambda x: x / 1e9 # "linear" (default), "log", "dB"
    f11_real.unitx = "GHz"
    f11_real.title = "Fit of $\\text{Re}(S_{11})$"
    f11_real.labelx = "Frequency"
    f11_real.labely = "$\\text{Re}(S_{11})$"
    f11_real.show_initial_model = True
    f11_real.show_plot = True
    f11_real.show_pvalue = False
    f11_real.show_model = True
    f11_real.figure_size = (30, 24)

    f21_real.params = {
        "QL": (0, QL, None), 
        "f0": (0, f0, None),
        "k1": (0, k1, 100),
        "k2": (0, k2, 100),
        "A": (0, A, 100),
        "theta21": (-np.pi, theta21, np.pi),
    }
    f21_real.param_units = { "QL": "1", "Qi": "1", "f0": "Hz", "k1": "1", "k2": "1", "A": "1", "theta21": "1" }
    f21_real.derived_params = {
        "Qi": lambda par: par["QL"]["value"] * (1 + par["k1"]["value"] + par["k2"]["value"]),
    }
    f21_real.param_displayed_names = { 
        "Qi": "Q_i = (1 + k_1 + k_2) Q_L",
        "QL": "Q_L",
        "f0": "f_r", 
        "k1": "\\kappa_1",
        "k2": "\\kappa_2",
        "theta21": "\\theta_{21}",
    }
    f11_real.params = {
        "QL": f21_real.params["QL"], 
        "f0": f21_real.params["f0"],
        "k1": f21_real.params["k1"],
        "k2": f21_real.params["k2"],
        "A": f21_real.params["A"],
        "theta11": (-np.pi, theta11, np.pi), # TODO: should not need to subtract pi
    }
    f11_real.param_units = { "QL": "1", "f0": "Hz", "k1": "1", "k2": "1", "A": "1", "theta11": "1" }
    f11_real.param_displayed_names = { 
        "Qi": "Q_i",
        "QL": "Q_L",
        "f0": "f_r", 
        "k1": "\\kappa_1",
        "k2": "\\kappa_2",
        "theta11": "\\theta_{11}",
    }

    f21_imag = f21_real.deep_copy()
    f21_imag.datay = np.imag(S21)[crop[0]:crop[1]] # 1
    f21_imag.title = "Fit of $\\text{Im}(S_{21})$"
    f21_imag.labely = "$\\text{Im}(S_{21})$"
    f21_imag.model = lambda x, f0, QL, k1, k2, A, theta21: np.imag(model_S21(x, f0, QL, k1, k2, A, theta21))

    f11_imag = f11_real.deep_copy()
    f11_imag.datay = np.imag(S11)[crop[0]:crop[1]] # 1
    f11_imag.title = "Fit of $\\text{Im}(S_{11})$"
    f11_imag.labely = "$\\text{Im}(S_{11})$"
    f11_imag.model = lambda x, f0, QL, k1, k2, A, theta11: np.imag(model_S11(x, f0, QL, k1, k2, A, theta11))

    # ============================ FITTING ============================

    res = simultaneous_fit([f21_real, f21_imag, f11_real, f11_imag])

    # ============================ PLOTTING ============================

    f11_abs = f11_real.deep_copy()
    f11_abs.datay = np.abs(S11)[crop[0]:crop[1]] # 1
    f11_abs.title = "Fit of $|S_{11}|$"
    f11_abs.labely = "$|S_{11}|$"
    f11_abs.scaley = "dB"
    f11_abs.model = lambda x, f0, QL, k1, k2, A, theta11: np.abs(model_S11(x, f0, QL, k1, k2, A, theta11))

    f11_phase = f11_real.deep_copy()
    f11_phase.datay = np.angle(S11)[crop[0]:crop[1]] # 1
    f11_phase.title = "Fit of $\\text{Arg}(S_{11})$"
    f11_phase.labely = "$\\text{Arg}(S_{11})$"
    f11_phase.model = lambda x, f0, QL, k1, k2, A, theta11: np.angle(model_S11(x, f0, QL, k1, k2, A, theta11))

    if plot_S11_real == True: 
        f11_real.show_plot = True
        f11_real.plot(res)
    if plot_S11_imag == True: 
        f11_imag.show_plot = True
        f11_imag.plot(res)
    if plot_S11_abs == True: 
        f11_abs.show_plot = True
        f11_abs.plot(res)
    if plot_S11_phase == True: 
        f11_phase.show_plot = True
        f11_phase.plot(res)
    if isinstance(plot_S11_real, str): 
        f11_real.show_plot = False
        f11_real.file_name = plot_S11_real
        f11_real.plot(res)
    if isinstance(plot_S11_imag, str): 
        f11_imag.show_plot = False
        f11_imag.file_name = plot_S11_imag
        f11_imag.plot(res)
    if isinstance(plot_S11_abs, str): 
        f11_abs.show_plot = False
        f11_abs.file_name = plot_S11_abs
        f11_abs.plot(res)
    if isinstance(plot_S11_phase, str): 
        f11_phase.show_plot = False
        f11_phase.file_name = plot_S11_phase
        f11_phase.plot(res)

    f21_abs = f21_real.deep_copy()
    f21_abs.datay = np.abs(S21)[crop[0]:crop[1]] # 1
    f21_abs.title = "Fit of $|S_{21}|$"
    f21_abs.labely = "$|S_{21}|$"
    f21_abs.scaley = "dB"
    f21_abs.model = lambda x, f0, QL, k1, k2, A, theta21: np.abs(model_S21(x, f0, QL, k1, k2, A, theta21))

    f21_phase = f21_real.deep_copy()
    f21_phase.datay = np.angle(S21)[crop[0]:crop[1]] # 1
    f21_phase.title = "Fit of $\\text{Arg}(S_{21})$"
    f21_phase.labely = "$\\text{Arg}(S_{21})$"
    f21_phase.model = lambda x, f0, QL, k1, k2, A, theta21: np.angle(model_S21(x, f0, QL, k1, k2, A, theta21))

    if plot_S21_real == True: 
        f21_real.show_plot = True
        f21_real.plot(res)
    if plot_S21_imag == True: 
        f21_imag.show_plot = True
        f21_imag.plot(res)
    if plot_S21_abs == True: 
        f21_abs.show_plot = True
        f21_abs.plot(res)
    if plot_S21_phase == True: 
        f21_phase.show_plot = True
        f21_phase.plot(res)
    if isinstance(plot_S21_real, str): 
        f21_real.show_plot = False
        f21_real.file_name = plot_S21_real
        f21_real.plot(res)
    if isinstance(plot_S21_imag, str): 
        f21_imag.show_plot = False
        f21_imag.file_name = plot_S21_imag
        f21_imag.plot(res)
    if isinstance(plot_S21_abs, str): 
        f21_abs.show_plot = False
        f21_abs.file_name = plot_S21_abs
        f21_abs.plot(res)
    if isinstance(plot_S21_phase, str): 
        f21_phase.show_plot = False
        f21_phase.file_name = plot_S21_phase
        f21_phase.plot(res)

    return res