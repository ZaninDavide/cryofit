import numpy as np
from fitter.Fitter import Fitter

def simultaneous_fit(fitters):
    """
    Perform simultaneous fitting of multiple Fitter objects.
    
    parameters
     - fitters: list of Fitter objects to be fitted simultaneously
    """
    for f in fitters:
        if not isinstance(f, Fitter):
            raise ValueError("All elements in fitters must be instances of the Fitter class.")

    mfitter = Fitter()
    mfitter.datax = np.concat([f.datax for f in fitters])
    mfitter.datay = np.concat([f.datay for f in fitters])
    mfitter.sigmay = np.concat([f.sigmay for f in fitters])

    chunks = []
    cur = 0
    for f in fitters:
        chunks.append((cur, cur + len(f.datax)))
        cur += len(f.datax)
    
    mfitter.params = dict()
    for f in fitters:
        for key, value in f.params.items(): 
            if key not in mfitter.params:
                mfitter.params[key] = value
            elif value != mfitter.params[key]:
                raise ValueError(f"Parameter {key} has different values in different fitters.")
                
    mfitter.derived_params = dict()
    for f in fitters:
        for key, value in f.derived_params.items(): 
            if key not in mfitter.derived_params:
                mfitter.derived_params[key] = value
    
    # Assumes that all data is fed at once to the model, which is true in the case of Fitter.fit()
    def model(datax, **params):
        datay = np.zeros(len(datax))
        for i, ch in enumerate(chunks):
            iparams = dict()
            for key, value in params.items():
                if key in fitters[i].params:
                    iparams[key] = value            
            datay[ch[0]:ch[1]] = fitters[i].model(datax[ch[0]:ch[1]], **iparams)
        return datay
    
    par_names = ', '.join(mfitter.params.keys())
    par_names_and_values = ', '.join([f"{key}={key}" for key in mfitter.params.keys()])
    func_code = f'''def dynamic_func(x, {par_names}): return model(x, {par_names_and_values})'''
    local_namespace = {}
    exec(func_code, {'model': model}, local_namespace)
    named_model = local_namespace['dynamic_func']
    mfitter.model = named_model
    
    return mfitter.fit()