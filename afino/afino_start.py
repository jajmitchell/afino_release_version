
import os
import datetime
import numpy as np
import pickle as pkl

from afino_code.afino.afino_series import AfinoSeries, prep_series
from afino_code.afino.afino_model_comparison import model_comparison
from afino_code.afino import afino_spectral_models
import matplotlib.pyplot as plt


def analyse_series(times, flux, description=None, low_frequency_cutoff=None, savedir=None, overwrite_gauss_bounds=None,
                       overwrite_extra_gauss_bounds=None, use_json=False, model_ids=[0,1,2], save=False):
    """
    Analyse a single, generic timeseries using the AFINO model comparison code.
    
    Parameters
    ----------

    times : ndarray
        an array of times
    flux : ndarray
        an array of data points
    description : string, optional
        a string descriptor of the analysis run, that is incorporated into output filenames
    low_frequency_cutoff : float, optional
        specifies a frequency above which the input Fourier spectrum is not analysed
    savedir : string, optional
        specifies a directory for output save files
    use_json : bool
        If set to True, saves analysis output in JSON format. If False, pickle format is used.
        Default is True.
    save : bool
        If set to True, saves the analysis results. Default is False.

    """

    ts = AfinoSeries(times,flux)
    #first need to apply a window function 
    sig_apodized=prep_series(ts)
    #now perform model comparison
    if not description:
        description = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') 
   
    results, analysis_summary=model_comparison(sig_apodized, description=description,
                                    low_frequency_cutoff=low_frequency_cutoff,
                                    overwrite_gauss_bounds=overwrite_gauss_bounds,

                                    overwrite_extra_gauss_bounds=overwrite_extra_gauss_bounds,
                                    use_json=use_json,
                                    model_ids=model_ids,
                                    savedir=savedir,
                                    save_status=save)

    af_qpp_det, af_qpp = create_generic_summary_plot(ts,
                                analysis_summary,
                                description,
                                low_frequency_cutoff=low_frequency_cutoff,
                                savedir=savedir, save=save)
    
    return af_qpp_det, af_qpp
   
    
    
def create_generic_summary_plot(ts, analysis_summary, description, low_frequency_cutoff=None, savedir=None, save=False):

    import seaborn as sns
    sns.set_style('ticks', {'xtick.direction': 'in', 'ytick.direction': 'in'})
    sns.set_context('paper')

    if savedir is None and save is True:
        os.makedirs(os.path.expanduser('~/afino_repository/plots/'), exist_ok=True)

    npanels = len(analysis_summary) + 1

    plt.figure(1, figsize=(6 * npanels, 5))
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95)

    plt.subplot(1, npanels, 1)
    plt.tick_params(labelsize=12)
    plt.plot(ts.SampleTimes.time, ts.data)

    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Intensity (arb.)')
    plt.title('Input signal', fontsize=16)

    S = []

    for i, key in enumerate(analysis_summary.keys()):
        plt.subplot(1, npanels, i + 2)
        plt.tick_params(labelsize=12)

        plt.loglog(analysis_summary[key]['frequencies'], analysis_summary[key]['power'], label='data')
        plt.loglog(analysis_summary[key]['frequencies'], analysis_summary[key]['best_fit_power_spectrum'], label='best fit')
        plt.legend(fontsize=14)

        plt.xlabel('frequency (Hz)', fontsize=12)
        plt.ylabel('Fourier power', fontsize=12)
        plt.title('PSD - Model ' + str(analysis_summary[key]['ID']), fontsize=12)

        plt.xlim([1e-5, 1e2])
        plt.text(2e-5, 1e-2, r'$\alpha=%4.2f$' % analysis_summary[key]['params'][1], fontsize=14)
        plt.text(2e-3, 1.0, r'$\chi^2=%4.2f$' % analysis_summary[key]['rchi2'], fontsize=14)

        S.append(analysis_summary[key]['BIC'])

        if low_frequency_cutoff:
            plt.axvline(low_frequency_cutoff)

        if analysis_summary[key]['model'] in ['pow_const_gauss', 'pow_const_2gauss']:

            f0 = np.exp(analysis_summary[key]['params'][4])
            period0 = 1 / f0

            plt.axvline(f0, color='red', linestyle='--')

            if analysis_summary[key]['model'] == 'pow_const_2gauss':
                f1 = np.exp(analysis_summary[key]['params'][7])
                period1 = 1 / f1
                plt.axvline(f1, color='red', linestyle='--')

    # -------------------------
    # Detection logic (FIXED)
    # -------------------------
    S = np.array(S)

    if len(S) >= 3:
        delta0 = abs(S[0] - S[1])
        delta1 = abs(S[1] - S[2])
        af_qpp_det = (delta0 > 10) and (delta1 > 10)
    else:
        af_qpp_det = False

    # -------------------------
    # Always extract highest period (regardless of detection)
    # -------------------------
    best_periods = []

    for key, summary in analysis_summary.items():
        model = summary['model']

        if model in ['pow_const_gauss', 'pow_const_2gauss']:
            best_periods.append(1 / np.exp(summary['params'][4]))

            if model == 'pow_const_2gauss':
                best_periods.append(1 / np.exp(summary['params'][7]))

    af_qpp = max(best_periods) if len(best_periods) > 0 else None

    # -------------------------
    # Save section (unchanged logic)
    # -------------------------
    savefilename = 'summary_plot_' + description + '.pdf'

    if savedir and save is True:
        plt.savefig(os.path.join(savedir, savefilename))

        file = description.replace('-', '')
        pkl_file = os.path.join(os.path.dirname(savedir), 'save', 'variables_' + description + '.pkl')

        with open(pkl_file, "rb") as f:
            data = pkl.load(f)

        data.update({
            "afino_best_model": None,
            "afino_best_period": best_periods,
            "afino_qpp_detected": [af_qpp_det, af_qpp],
        })

        with open(pkl_file, "wb") as f:
            pkl.dump(data, f)

    elif save is True:
        plt.savefig(os.path.join(os.path.expanduser('~/afino_repository/plots/'), savefilename))

    plt.show()
    plt.close()

    return af_qpp_det, af_qpp