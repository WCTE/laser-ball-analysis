#!/usr/bin/env python

"""
PMT Charge Distribution Fitting Script
Fits PMT response using the Bellamy model with lmfit
"""

import re
import os
import sys
import glob
import math
import time
import logging
from platform import python_version
from typing import List, Dict, Any, Optional
import copy
import argparse

from scipy.special import erf, erfc, erfcx
import numpy as np
import pandas as pd
import awkward as ak
import scipy.stats as stats
import lmfit
from lmfit import Model, minimize, Parameters, report_fit
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tabulate import tabulate
import pickle

import EventDisplay
from wcte_pmt_mapping import PMTMapping as mapping

import ROOT
import uproot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_RUN_NUM = "2307"
DEFAULT_MODEL = "bellamy"

# Plot configuration
PLOT_CONFIG = {
    'bins': (-50, 400, 1),
    'draw_function_bins': (-50, 400, 1),
    'ylim_bottom': 1,
    'ylim_top': 1e7,
    'ylim_top_gaus': 1e7,
    'xlim_bottom': -50,
    'xlim_top': 400,
    'xlim_bottom_gaus': -50,
    'xlim_top_gaus': 400,
    'subplot_shape': (4, 5),
    'subplot_size': (24, 14),
}

# Initial fit parameters
INITIAL_PARAMS = {
    'N': {'value': 3e3},
    'Q0': {'value': 0},
    's0': {'value': 2},
    'Q1': {'value': 130},
    's1': {'value': 40},
    'alpha': {'value': 0.008},
    'mu': {'value': 0.01},
    'w': {'value': 0.01},
}

INITIAL_GAUS_PARAMS = {
    'N': {'value': 5e5},
    'Q0': {'value': 0},
    's0': {'value': 2},
}

def gaussian(x: np.ndarray, Qn: float, sigman: float) -> np.ndarray:
    """Gaussian distribution for n photoelectrons."""
    # norm from paper
    norm = 1 / (sigman * np.sqrt(2 * np.pi))
    return norm * np.exp(-(x - Qn) ** 2 / (2 * sigman ** 2))


def pedestal_response(x: np.ndarray, N: float, Q0: float, s0: float) -> np.ndarray:
    signal = np.zeros_like(x, dtype=float)
    
    for n in range(1):
        Qn = Q0
        s_n = np.sqrt(s0 ** 2)
        if n == 0:
            gauss_term = gaussian(x, Q0, s0)
        signal += gauss_term
    
    if np.any(np.isnan(N * signal)):
        print(f"DEBUG: NaN generated in output. Inputs: N={N}, s0={s0}, ...")
    if np.any(np.isinf(N * signal)):
        print(f"DEBUG: Inf detected! ")
    return N * signal 

def gauss_response(x: np.ndarray, N: float, Q0: float, s0: float, Q1: float, 
                 s1: float, mu: float, w: float) -> np.ndarray:
    signal = np.zeros_like(x, dtype=float)
    
    for n in range(3):
        Qn = Q0 + n * Q1
        s_n = np.sqrt(s0 ** 2 + n * s1 ** 2)
        
        poisson = poisson_term(mu, n)
        
        if n == 0:
            gauss_term = gaussian(x, Q0, s0)
        else:
            gauss_term = gaussian(x, Qn, s_n)
        
        signal += poisson * ((1 - w) * gauss_term)
    
    if np.any(np.isnan(N * signal)):
        print(f"DEBUG: gaus NaN generated in output. Inputs: N={N}, s0={s0}, ...")
    if np.any(np.isinf(N * signal)):
        print(f"DEBUG: gaus Inf detected! ")
    return N * signal 

def poisson_term(mu: float, n: int) -> float:
    """Poisson probability for n photoelectrons given mean mu."""
    return mu ** n * np.exp(-mu) / math.factorial(n)

def background_response(x: np.ndarray, N: float, Q0: float, s0: float, Q1: float, 
                 s1: float, mu: float, w: float, alpha: float) -> np.ndarray:
    signal = np.zeros_like(x, dtype=float)
    
    for n in range(3):
        Qn = Q0 + n * Q1
        s_n = np.sqrt(s0 ** 2 + n * s1 ** 2)
        
        poisson = poisson_term(mu, n)
        
        # mask 
        threshold = Q0
        mask = x > threshold
        theta = mask.astype(int) 
        
        exp_term = theta * alpha / 2 * np.exp(-alpha * (x - Qn - alpha * s_n ** 2 / 2))
        erf_arg = x - Qn - s_n ** 2 * alpha
        erf1 = math.erf(abs(Q0 - Q1 - s_n ** 2 * alpha) / (s_n * np.sqrt(2)))
        erf2 =  np.sign(erf_arg) * np.vectorize(math.erf)(abs(erf_arg) / (s_n * np.sqrt(2)))
        
        ig_n_term = exp_term * (erf1 + erf2) 
        signal += poisson * (w * ig_n_term)
    
    if np.any(np.isnan(N * signal)):
        print(f"DEBUG: background NaN generated in output. Inputs: N={N}, s0={s0}, ...")
    if np.any(np.isinf(N * signal)):
        print(f"DEBUG: background Inf detected! ")
    return N * signal 

def pmt_response(x: np.ndarray, N: float, Q0: float, s0: float, Q1: float, 
                 s1: float, mu: float, w: float, alpha: float) -> np.ndarray:
    """
    PMT response model combining Gaussian and exponential terms.
    
    Parameters:
        x: Charge values
        N: Normalization factor
        Q0: Pedestal charge
        s0: Pedestal width
        Q1: Single photoelectron charge
        s1: Single photoelectron width
        mu: Mean number of photoelectrons
        w: Weight for type 2 process
        alpha: Exponential decay parameter
    """
    signal = np.zeros_like(x, dtype=float)
    
    for n in range(3):
        Qn = Q0 + n * Q1
        s_n = np.sqrt(s0 ** 2 + n * s1 ** 2)
        
        poisson = poisson_term(mu, n)
        
        if n == 0:
            gauss_term = gaussian(x, Q0, s0)
        else:
            gauss_term = gaussian(x, Qn, s_n)
        
        # mask 
        threshold = Q0
        mask = x > threshold
        theta = mask.astype(int) 
        exp_term = theta * alpha / 2 * np.exp(-alpha * (x - Qn - alpha * s_n ** 2 / 2))
        
        erf_arg = x - Qn - s_n ** 2 * alpha
        erf1 = math.erf(abs(Q0 - Q1 - s_n ** 2 * alpha) / (s_n * np.sqrt(2)))
        erf2 =  np.sign(erf_arg) * np.vectorize(math.erf)(abs(erf_arg) / (s_n * np.sqrt(2)))
        
        ig_n_term = exp_term * (erf1 + erf2)
        signal += poisson * ((1 - w) * gauss_term + w * ig_n_term)
    
    if np.any(np.isnan(N * signal)):
        print(f"DEBUG: NaN generated in output. Inputs: N={N}, s0={s0}, ...")
    if np.any(np.isinf(N * signal)):
        print(f"DEBUG: Inf detected! ")
    return N * signal 


def calculate_r_squared(y: np.ndarray, result) -> float:
    """Calculate R-squared value for fit quality assessment."""
    residuals = result.residual
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def create_pmt_response_model() -> lmfit.Model:
    """Create and configure the PMT response model."""
    model = lmfit.Model(pmt_response)
    params = model.make_params(**{k: v['value'] for k, v in INITIAL_PARAMS.items()})
    return model, params

def create_pedestal_model() -> lmfit.Model:
    """Create and configure the Gaussian model."""
    model = lmfit.Model(pedestal_response)
    params = model.make_params(**{k: v['value'] for k, v in INITIAL_GAUS_PARAMS.items()})
    return model, params

def create_gaussian_model() -> lmfit.Model:
    """Create and configure the Gaussian model."""
    model = lmfit.Model(gauss_response)
    params = model.make_params(**{k: v['value'] for k, v in INITIAL_GAUS_PARAMS.items()})
    return model, params

def create_background_model() -> lmfit.Model:
    """Create and configure the Gaussian model."""
    model = lmfit.Model(background_response)
    params = model.make_params(**{k: v['value'] for k, v in INITIAL_PARAMS.items()})
    return model, params

def charge_fit(data: pd.DataFrame, run, model: str = "bellamy", 
               remove_sat: bool = False, verbose: bool = False, card_id: int = -99) -> List[List[Any]]:
    """
    Fit charge distributions for all PMTs in the dataset.
    
    Args:
        data: DataFrame with 'card', 'channel', 'charge' columns
        run: Run number of data
        model: Model type to use for fitting
        remove_sat: Whether to remove saturated channels
        verbose: Enable debug printing
    
    Returns:
        List of fit results for each PMT
    """
    FIGURE_DIR = 'figures/'+f'{run}'  
    directory = Path(FIGURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    
    no_row = 19
    result_array = [[None] * 11 for _ in range(0,no_row,1)]
    gaus_result_array = [[None] * 6 for _ in range(0,no_row,1)]
    
    fig, ax = plt.subplots(*PLOT_CONFIG['subplot_shape'], 
                           figsize=PLOT_CONFIG['subplot_size'])
    fig_gaus, ax_gaus = plt.subplots(*PLOT_CONFIG['subplot_shape'], 
                           figsize=PLOT_CONFIG['subplot_size'])
    fig_gaus_res, ax_gaus_res = plt.subplots(*PLOT_CONFIG['subplot_shape'], 
                           figsize=PLOT_CONFIG['subplot_size'])
    fig_res, ax_res = plt.subplots(*PLOT_CONFIG['subplot_shape'], 
                           figsize=PLOT_CONFIG['subplot_size'])
    fig_chi, ax_chi = plt.subplots(*PLOT_CONFIG['subplot_shape'], 
                           figsize=PLOT_CONFIG['subplot_size'])
    
    pedestal_model, pedestal_params_template = create_pedestal_model()
    pmt_model, params_template = create_pmt_response_model()
    gaus_model, gaus_params_template = create_gaussian_model()
    background_model, background_params_template = create_background_model()
    
    counter = 0
    for row in range(0,no_row,1):
        counter += 1
        card = card_id
        channel = data['channel_ids'][row]
        charges = data["charges"][data["channel_ids"]==row]
        
        # Create histogram
        bins = np.arange(*PLOT_CONFIG['bins'])
        bin_centres = bins[:-1] + (bins[1] - bins[0]) / 2
        hist, _ = np.histogram(charges, bins=bins)
        
        # Plot histogram
        ax[channel // 5, channel % 5].cla()
        ax[channel // 5, channel % 5].hist(charges, bins=bins)
        ax[channel // 5, channel % 5].set_title(f"Channel {channel}")
        ax[channel // 5, channel % 5].set_ylim(PLOT_CONFIG['ylim_bottom'], 
                                                PLOT_CONFIG['ylim_top'])
        ax[channel // 5, channel % 5].set_yscale('log')
        ax[channel // 5, channel % 5].set_xlim(PLOT_CONFIG['xlim_bottom'], 
                                                PLOT_CONFIG['xlim_top'])

        ax_gaus[channel // 5, channel % 5].cla()
        ax_gaus[channel // 5, channel % 5].hist(charges, bins=bins)
        ax_gaus[channel // 5, channel % 5].set_title(f"Channel {channel}")
        ax_gaus[channel // 5, channel % 5].set_ylim(PLOT_CONFIG['ylim_bottom'], 
                                                PLOT_CONFIG['ylim_top_gaus'])
        ax_gaus[channel // 5, channel % 5].set_yscale('log')

        n, bins, patches = ax[channel // 5, channel % 5].hist(charges, bins=bins)
        mode_bin_midpoint = (bins[np.argmax(n)] + bins[np.argmax(n) + 1]) / 2
 
        # 1 pe position
        low, high = 150, 250
        mask = (bins[:-1] >= low) & (bins[:-1] < high)
        filtered_n = n[mask]
        max_bin_index = np.argmax(filtered_n)
        onepe_peak_height = np.max(filtered_n)
        filtered_bins_left = bins[:-1][mask]
        filtered_bins_right = bins[1:][mask]
        max_bin_midpoint = (filtered_bins_left[max_bin_index] + filtered_bins_right[max_bin_index]) / 2
        max_bin_midpoint = 140

        # ped peak position
        low, high = 0, 75
        mask = (bins[:-1] >= low) & (bins[:-1] < high)
        filtered_n = n[mask]
        ped_peak_height = np.max(filtered_n)

        # N of histo
        bin_width = 1
        integral_tail = np.sum(n[-50:] * bin_width)
        integral = np.sum(n * bin_width)
        integral_ped = np.sum(n[25:-375] * bin_width)
        integral_1pe = np.sum(n[75:] * bin_width)
        integral_ratio = integral_ped / integral_1pe

        # set seeds and ranges 
        temp_params = copy.deepcopy(INITIAL_PARAMS)
        temp_params['N']['value'] = integral  
        INITIAL_GAUS_PARAMS['N']['value'] = integral  
        temp_params['Q0']['value'] = mode_bin_midpoint  
        temp_params['Q1']['value'] = max_bin_midpoint  

        if model == "bellamy":
            try:
             r2 = 0  
             count = 0
             while r2 < 0.992 and count < 1:
                # Perform fit
                params = pedestal_model.make_params(**INITIAL_GAUS_PARAMS, method='brute', max_nfev=100000)
                result = pedestal_model.fit(hist[40:390], params, x=bin_centres[40:390])
                
                if result.success:
                    # Extract fit parameters
                    fit_params = {
                        'N': result.best_values['N'],
                        'Q0': result.best_values['Q0'],
                        's0': result.best_values['s0'],
                    }
                    
                    temp_params['Q0']['value'] = fit_params['Q0']
                    temp_params['s0']['value'] = fit_params['s0']
                    
                    draw_gaus = [fit_params['N'], fit_params['Q0'], fit_params['s0']]
                    gaus_line = pedestal_response(bin_centres, *draw_gaus)
                    ax_gaus[channel // 5, channel % 5].set_xlim(PLOT_CONFIG['xlim_bottom_gaus'], PLOT_CONFIG['xlim_top_gaus'])
                    
                    r2 = calculate_r_squared(hist, result)
                    chi = result.redchi
                    
                    logger.info(f"Card {card} Channel {channel}: R² = {r2:.4f}, χ² = {chi:.2f}")
                    
                    # Store results
                    fit_data = [card, channel] + list(fit_params.values()) + [r2, chi]
                    
                    # Plot fit
                    plot_result = [fit_params[k] for k in ['N', 'Q0', 's0']]
                    ax_gaus[channel // 5, channel % 5].plot(bin_centres, pedestal_response(bin_centres, *plot_result),label=f'fitted line')
                    ax_gaus[channel // 5, channel % 5].legend(loc='best')
                    count += 1
                    
                else:
                    logger.warning(f"Fit failed for Card {card} Channel {channel}")
                    fit_data = [card, channel] + [None] * 10
                    count += 1
                    
            except Exception as e:
                logger.error(f"Error fitting Card {card} Channel {channel}: {e}")
                fit_data = [card, channel] + [None] * 10
            
            gaus_result_array[row] = fit_data
            fig_gaus.savefig(os.path.join(FIGURE_DIR, f"Run_{DEFAULT_RUN_NUM}_gaus_card_{card}.png"))
            fig_gaus_res.savefig(os.path.join(FIGURE_DIR, f"Run_{DEFAULT_RUN_NUM}_gaus_card_{card}_res.png"))
            fig_chi.savefig(os.path.join(FIGURE_DIR, f"Run_{DEFAULT_RUN_NUM}_gaus_card_{card}_chi.png"))
            
            r2 = 0  
            count = 0
            while r2 < 0.992 and count < 1:
                # Perform fit
                params = pmt_model.make_params(**temp_params, method='brute', max_nfev=100000)
                params['Q0'].set(vary=False)
                params['s0'].set(vary=False)
                fit_start = 50 + params['Q0'].value + 15
                result = pmt_model.fit(hist[round(fit_start):], params, x=bin_centres[round(fit_start):])
                
                if result.success:
                    # Extract fit parameters
                    fit_params = {
                        'N': result.best_values['N'],
                        'Q0': result.best_values['Q0'],
                        's0': result.best_values['s0'],
                        'Q1': result.best_values['Q1'],
                        's1': result.best_values['s1'],
                        'mu': result.best_values['mu'],
                        'w': result.best_values['w'],
                        'alpha': result.best_values['alpha'],
                    }
                    temp_params = copy.deepcopy(fit_params)
                    contribution_params = copy.deepcopy(fit_params)
        
                    del contribution_params['alpha']

                    #ax[channel // 5, channel % 5].plot(draw_function_bin_centres, gauss_response(draw_function_bin_centres, **contribution_params),label=f'gaussian')
                    #ax[channel // 5, channel % 5].plot(draw_function_bin_centres, background_response(draw_function_bin_centres, **temp_params),label=f'background')
                    
                    r2 = calculate_r_squared(hist, result)
                    chi = result.chisqr/len(bins)
                    
                    logger.info(f"Card {card} Channel {channel}: R² = {r2:.4f}, χ² = {chi:.2f}")
                    
                    MapCode = mapping()
                    slot_position = []
                    slot_position.append(MapCode.get_slot_pmt_pos_from_card_pmt_chan(int(card), int(channel)))
                    # Store results
                    fit_data = [card, channel] + list(fit_params.values()) + [r2, chi] + [str(slot_position[0][0]), str(slot_position[0][1])]
                    
                    # Plot fit
                    plot_result = [fit_params[k] for k in ['N', 'Q0', 's0', 'Q1', 's1', 'mu', 'w', 'alpha']]
                    ax[channel // 5, channel % 5].plot(bin_centres, pmt_response(bin_centres, *plot_result),label=f'fitted line')
                    ax[channel // 5, channel % 5].legend(loc='best')
                    count += 1
                    
            result_array[row] = fit_data
            fig.savefig(os.path.join(FIGURE_DIR, f"Run_{DEFAULT_RUN_NUM}_bellamy_card_{card}.png"))
            fig_res.savefig(os.path.join(FIGURE_DIR, f"Run_{DEFAULT_RUN_NUM}_bellamy_card_{card}_res.png"))
            fig_chi.savefig(os.path.join(FIGURE_DIR, f"Run_{DEFAULT_RUN_NUM}_bellamy_card_{card}_chi.png"))
    
    result_arrays = [gaus_result_array, result_array]
    PICKLE_DIR = 'pickle_dir/'+f'{run}'  
    directory = Path(PICKLE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"pickle_dir/{run}/results_array{card}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(result_arrays, f)
    return result_arrays


def fit_file(base_path, run_num: str, card: str, out_file, remove_sat: bool = False) -> pd.DataFrame:
    """
    Process all files for a given run and perform fits.
    
    Args:
        base_path: Directory containing input ROOT files
        run_num: Run number identifier
        model: Model type to use
        remove_sat: Whether to remove saturated channels
    
    Returns:
        DataFrame with all fit results
    """
    file_pattern = f"{base_path}/{run_num}/processed_hits.root"
    file_list = sorted(glob.glob(file_pattern))
    
    if not file_list:
        logger.warning(f"No files found matching pattern: {file_pattern}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(file_list)} files to process")
    
    fit_arr = []
    for file_path in file_list:
        logger.info(f"Fitting {file_path}")
        data = []
        with uproot.open(file_path) as file:
            for tree_name, tree in file.items():
                card_id = int(re.search(r'\d+', tree_name).group())
                if int(card_id) != int(card): continue 
                branches = ["channel_ids", "charges", "pmt_positions"]
                data = tree.arrays(branches, library="np")
                fit_data = charge_fit(data, run_num, model="bellamy", remove_sat=remove_sat, card_id=str(card_id))
                fit_arr.extend(fit_data)

    # Create results DataFrame
    fig_gain, ax_gain = plt.subplots(1, 1)
    fig_mu, ax_mu = plt.subplots(1, 1)
    model = "bellamy"
    card = str(card)
    if model == "bellamy":
        columns_gaus = ['card', 'channel', 'N', 'Q0', 's0', 'r2', 'chi/NDF']
        columns = ['card', 'channel', 'N', 'Q0', 's0', 'Q1', 's1', 'mu', 'w', 'alpha', 'r2', 'chi/NDF', 'slot', 'position']
        fit_frame_gaus = pd.DataFrame.from_records(fit_arr[0], columns=columns_gaus)
        fit_frame = pd.DataFrame.from_records(fit_arr[1], columns=columns)
        column_to_plot = 'Q1'
        mean_gain = fit_frame['Q1'].mean()
        threshold = 0.20 * 140
        outliers = fit_frame[abs(fit_frame['Q1'] - mean_gain) > threshold]
        if not outliers.empty:
            with open(f'{out_file}/{run_num}/outlier.txt', 'a') as f:
                f.write(outliers.to_string(index=False))
                f.write('\n')


        PMT = ((fit_frame['card'].astype(int) - 1) * 19) + fit_frame['channel'].astype(int)
        gain = fit_frame['Q1'] - fit_frame['Q0']
        ax_gain.scatter(PMT, gain, color='red', alpha=0.6, s=10)  
        ax_gain.set_ylim(100, 200)
        ax_gain.text(
            x=0.05,
            y=0.95,
            s=f'Mean Gain: {mean_gain:.2f}',
            transform=ax_gain.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)  # Optional background box
        )
        ax_gain.set_title(f"Gain vs. PMT {card}")
        ax_gain.set_xlabel('PMT')
        ax_gain.set_ylabel('Gain')
        fig_gain.savefig(os.path.join(FIGURE_DIR, f"Run_{DEFAULT_RUN_NUM}_CARD_"+card+"_gains.png"))
        
        # mu variation
        ax_mu.scatter(PMT, fit_frame['mu'], color='red', alpha=0.6, s=10)  
        ax_mu.set_title(f'mu vs. PMT {card}')
        ax_mu.set_xlabel('PMT')
        ax_mu.set_ylabel('mu')
        fig_mu.savefig(os.path.join(FIGURE_DIR, f"Run_{DEFAULT_RUN_NUM}_CARD_"+card+"_mu.png"))

        figures = {}
        axes = {} 
    
        j = 0                                                                                                
        col = ['N', 'Q0', 's0', 'Q1', 's1', 'mu', 'w', 'alpha', 'chi/NDF']
        for i in col: 
           figures[i], axes[i] = plt.subplots(1, 1)
    
           axes[i].scatter(fit_frame[i], gain, color='red', alpha=0.6, s=10)
           axes[i].set_title(f'{i} vs. gain')
           axes[i].set_xlabel(f'{i}')
           axes[i].set_ylabel('gain')
       
           if i == 'chi/NDF': 
              save = 'chi'
              filename = f"Run_{DEFAULT_RUN_NUM}_CARD_"+card+f"_gainVS{save}.png"   
              figures[i].savefig(os.path.join(FIGURE_DIR, filename))                                        
           else:
              filename = f"Run_{DEFAULT_RUN_NUM}_CARD_"+card+f"_gainVS{i}.png"
              figures[i].savefig(os.path.join(FIGURE_DIR, filename))                                        
              plt.close(figures[i])
           j += 1


    # Display results
    print(tabulate(fit_frame_gaus, headers='keys', tablefmt='psql', showindex=False))
    print(tabulate(fit_frame, headers='keys', tablefmt='psql', showindex=False))
    
    logger.info("Fitting completed successfully")
    return fit_frame


def main(base_path, run, card, out_file):
    """Main entry point."""
    
    logger.info(f"Starting fit for Run {run} with model bellemy")
    fit_file(base_path, run, card, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PMT waveforms from ROOT files and write hits to a new ROOT file."
    )
    parser.add_argument("--base-path",   type=str, required=True,
                        help="Directory containing input ROOT files")
    parser.add_argument("--run",         type=int, required=True,
                        help="Run number")
    parser.add_argument("--card",         type=int, required=True,
                        help="Card number")
    parser.add_argument("--out-file",   type=str, required=True,
                        help="Path to output")
    
    args = parser.parse_args()
    main(**vars(args))
