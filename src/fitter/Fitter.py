import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy import stats

from utils import number_to_text 
from utils import error_propagation 


class Fitter():
    """
    Fitter has the following parameters:
     - datax (numpy.ndarray)
     - datay (numpy.ndarray)
     - sigmay (numpy.ndarray)
     - model (function)    
     - params (dict)
     - derived_params (dict)
     - param_units (dict)
     - param_displayed_names (dict)
     - title (str)
     - labelx (str)
     - labely (str)
     - unitx (str)
     - unity (str)
     - scaley (str)
     - scalex (str)
     - number_of_errorbars (int)
     - show_plot (bool)
     - show_initial_model (bool)
     - show_model (bool)
     - show_errorbars (bool)
     - show_pvalue (bool)
     - figure_size (tuple)
     - file_name (str)

    ## `fit(self)`
    Returns a dictionary with fields
     - "chi2": non-reduced chi squared
     - "ndof": number of degrees of freedom
     - "reduced_chi2": reduced chi squared
     - "pvalue": p-value of the fit
     - "params": dictionary where the keys are the names of the parameters and the values are dictionary `{"value": float, "sigma": float}`
     - "derived_params": like above

    ## `plot_fit(self)`
    Returns a dictionary with the same fields as `fit` plus the field "plot" which is the matplotlib plotting object.
    
    ## `plot(self, res)`
    Plot result of a call to fit().

    ## Conventions
     - All arrays are numpy arrays
    """

    datax = np.array([])
    datay = np.array([])
    sigmay = np.array([])
    model = None
    
    params = {}
    param_units = {}
    param_displayed_names = {}
    derived_params = {}
    params_displayer = "box" # None, "box", "table" 
    
    title = "Title"
    labelx = "x"
    labely = "y"
    unitx = ""
    unity = ""

    scaley = "linear"
    scalex = "linear"

    number_of_errorbars = 50
    show_plot = False
    show_initial_model = False
    show_model = True
    show_errorbars = True
    show_pvalue = False
    figure_size = (20, 15)
    file_name = ""

    def fit(self):
        separated = separate_values_from_limits(self.params)
        
        least_squares = LeastSquares(self.datax, self.datay, self.sigmay, self.model)
        m = Minuit(least_squares, **separated["values"]) 
        
        for key, value in separated["limits"].items(): m.limits[key] = value
        
        m.migrad()
        m.hesse()

        final_params = {}
        for p, v, e in zip(m.parameters, m.values, m.errors):
            final_params[p] = {"value": v, "sigma": e}

        final_derived_params = {}
        for name, func in self.derived_params.items():
            final_derived_params[name] = {
                "value": func(final_params),
                "sigma": error_propagation(func, final_params)
            }

        return {
            "chi2": m.fval,
            "ndof": m.ndof,
            "reduced_chi2": m.fmin.reduced_chi2,
            "pvalue": float(1 - stats.chi2.cdf(m.fval, m.ndof)),
            "params": final_params,
            "derived_params": final_derived_params,
        }
    
    def plot(self, res):
        final_values = {}
        for key in self.params.keys():
            final_values[key] = res["params"][key]["value"]

        # SUBPLOTS
        fig, axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})
        fig.set_size_inches(self.figure_size[0] / 2.54, self.figure_size[1] / 2.54)
        first = axes[0]
        second = axes[1]

        # MAIN PLOT
        modelx = np.linspace(np.min(self.datax), np.max(self.datax), 2000)
        modely = [self.model(f, **final_values) for f in modelx]

        scalex_pass = lambda data: data
        scaley_pass = lambda data: data
        if callable(self.scalex): # if is function
            scalex_pass = self.scalex
        elif self.scalex == "dB":
            scalex_pass = lambda data: 20*np.log(data)
        if callable(self.scaley): # if is function
            scaley_pass = self.scaley
        elif self.scaley == "dB":
            scaley_pass = lambda data: 20*np.log(data)
        
        first.scatter(scalex_pass(self.datax), scaley_pass(self.datay), label="data", marker="o", facecolors="none", edgecolors="#1f73f0")
        if self.show_model:
            first.plot(scalex_pass(modelx), scaley_pass(modely), color="#f01f1f", label="fit")
        if self.show_initial_model:
            separated_initial = separate_values_from_limits(self.params)
            initialy = [self.model(f, **separated_initial["values"]) for f in modelx]
            first.plot(scalex_pass(modelx), scaley_pass(initialy), color="green", label="initial")

        # Axis
        if self.scalex == "log": first.set_xscale("log")
        if self.scaley == "log": first.set_yscale("log")

        final_unity = ("dB" if self.scaley == "dB" else "") + (self.unity if self.unity != "1" else "")
        labely_with_unit = f"{self.labely} [{final_unity}]" if (final_unity != "1" and final_unity != "") else self.labely
        first.set(ylabel=labely_with_unit)

        first.set_xticks(np.linspace(np.min(scalex_pass(self.datax)), np.max(scalex_pass(self.datax)), 25))
        first.set_xlim(scalex_pass(np.min(self.datax)), scalex_pass(np.max(self.datax)))
        first.set_xticklabels([''] * 25)
        first.grid(linestyle='--', linewidth=0.5)

        # Box
        texts = []

        for key in self.params.keys():
            par = res["params"][key]
            name = self.param_displayed_names[key] if key in self.param_displayed_names else key
            if name != "":
                texts.append(rf"${name}$ = ${number_to_text(par["value"], par["sigma"], self.param_units[key])}$")
        for key in self.derived_params.keys():
            par = res["derived_params"][key]
            name = self.param_displayed_names[key] if key in self.param_displayed_names else key
            if name != "":
                texts.append(rf"${name}$ = ${number_to_text(par["value"], par["sigma"], self.param_units[key])}$")
        
        if self.show_pvalue: texts.append(f"p-value = {res["pvalue"]:.2f}")
        text = "\n".join(texts)

        xaxis_min, xaxis_max = first.get_xlim()
        yaxis_min, yaxis_max = first.get_ylim()

        if self.params_displayer == "box":
            loc = 2
            if (yaxis_max + yaxis_min) / 2.0 < scaley_pass(self.datay[np.argmin(scalex_pass(self.datax))]): loc = 3
            anchored_text = AnchoredText(
                text, 
                loc = loc,
                pad = 0.5,
                borderpad = 0.5,
                prop = dict(fontsize = 12), 
            )
            first.add_artist(anchored_text)

        # Legend
        loc = 1
        if (yaxis_max + yaxis_min) / 2.0 < scaley_pass(self.datay[np.argmax(self.datax)]): loc = 4
        legend = first.legend(loc=loc, frameon=True, borderaxespad=0.8, fontsize=12)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_boxstyle("Square")

        # RESIDUALS PLOT
        residualsy = [ scaley_pass(self.datay[i]) - scaley_pass(self.model(self.datax[i], **final_values)) for i in range(len(self.datax)) ]
        zeroy = np.zeros(len(self.datax))
        N = int(len(self.datax) / self.number_of_errorbars)

        if N == 0 and self.show_errorbars:
            second.errorbar(
                scalex_pass(self.datax), 
                residualsy, 
                yerr = self.sigmay,
                ecolor = "#1f73f0",
                capsize = 5,
                fmt = '',
                linestyle=''
            )
        elif self.show_errorbars:
            second.errorbar(
                scalex_pass(self.datax[::N]), 
                residualsy[::N], 
                yerr = self.sigmay[::N],
                ecolor = "#1f73f0",
                capsize = 5,
                fmt = '',
                linestyle=''
            )
        second.plot(scalex_pass(self.datax), zeroy, color = "#f01f1f")
        second.scatter(scalex_pass(self.datax), residualsy, marker="o", facecolors="none", edgecolors="#1f73f0")

        # Axis
        if self.scalex == "log": second.set_xscale("log")

        final_unitx = ("dB" if self.scalex == "dB" else "") + (self.unitx if self.unitx != "1" else "")
        labelx_with_unit = f"{self.labelx} [{final_unitx}]" if (final_unitx != "1" and final_unitx != "") else self.labelx
        label_residuals_with_unit = f"Residuals [{final_unity}]" if (final_unity != "1" and final_unity != "") else "Residuals"
        second.set(xlabel=labelx_with_unit, ylabel=label_residuals_with_unit)

        second.set_xlim(*first.get_xlim())
        second.set_xticks(np.linspace(first.get_xlim()[0], first.get_xlim()[1], 25))
        plt.xticks(rotation=90)
        second.grid(linestyle='--', linewidth=0.5)

        # PARAMS TABLE

        tb = None
        cell_height = 0.08
        if self.params_displayer == "table":
            tb = first.table(cellText=[[self.title], [""], *map(lambda x: [x], texts), [""]], cellLoc="center", loc="top")
            tb[(3 + len(texts) - 1, 0)].visible_edges = ''
            plt.subplots_adjust(hspace = 0.05, top=(1 - cell_height * (len(texts) + 3) / 2.0))
            
            # font size
            tb.auto_set_font_size(False)
            tb.set_fontsize(12)
            # edges
            tb[(0, 0)].visible_edges = ''
            tb[(1, 0)].visible_edges = ''
            tb[(0, 0)].set_text_props(fontsize=14)

            for key, cell in tb.get_celld().items(): cell.set_height(cell_height)
        else:
            first.set_title(self.title, fontsize=14)
            
        # EXPORT

        if self.file_name != "": plt.savefig(self.file_name, bbox_inches='tight', dpi=200)
        if self.show_plot: plt.show() 
        else: plt.close()

        res["plot"] = plt
        return res
      
    def plot_fit(self):
        return self.plot(self.fit())
    
    def deep_copy(self):
        """
        Returns a deep copy of the Fitter object.
        """
        f = Fitter()
        f.datax = np.copy(self.datax)
        f.datay = np.copy(self.datay)
        f.sigmay = np.copy(self.sigmay)
        f.model = self.model
        f.params = self.params.copy()
        f.param_units = self.param_units.copy()
        f.param_displayed_names = self.param_displayed_names.copy()
        f.derived_params = self.derived_params.copy()
        f.title = self.title
        f.labelx = self.labelx
        f.labely = self.labely
        f.unitx = self.unitx
        f.unity = self.unity
        f.scaley = self.scaley
        f.scalex = self.scalex
        f.number_of_errorbars = self.number_of_errorbars
        f.show_plot = self.show_plot
        f.show_initial_model = self.show_initial_model
        f.show_model = self.show_model
        f.show_errorbars = self.show_errorbars
        f.show_pvalue = self.show_pvalue
        f.figure_size = self.figure_size
        f.file_name = self.file_name        
        return f

def separate_values_from_limits(params):
    params_values = {}
    params_limits = {}
    for key, value in params.items():
        params_values[key] = value[1]
        params_limits[key] = (value[0], value[2])
    return {"values": params_values, "limits": params_limits}

    
