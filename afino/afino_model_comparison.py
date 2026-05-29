import pickle
import os
import numpy as np
from afino.afino_main_analysis3 import main_analysis
from afino.afino_utils import model_string_from_id
from afino.afino_utils import save_afino_results

def model_comparison(ts,description=None,low_frequency_cutoff=None, overwrite_gauss_bounds = None,
                         overwrite_extra_gauss_bounds = None, use_json = True, model_ids = [0,1,2], savedir = None, save_status=False):
    """Initiate the comparison of different models to the Fourier power spectrum of a timeseries."""
    
    results = []
    for id in model_ids:
        model_string = model_string_from_id(id)
        result = main_analysis(ts, model=model_string, low_frequency_cutoff = low_frequency_cutoff, overwrite_gauss_bounds = overwrite_gauss_bounds,
                                   overwrite_extra_gauss_bounds = overwrite_extra_gauss_bounds)
        result['ID'] = id
        results.append(result)

    analysis_summary = save_afino_results(results, description = description, use_json = use_json, savedir = savedir, save=save_status)


    return results, analysis_summary
    
   