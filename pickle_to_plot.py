import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import EventDisplay
import matplotlib.colors as colors
import argparse
from scipy.optimize import curve_fit
from pathlib import Path

DEFAULT_RUN_NUM = "2307"

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def create_combined_histogram(base_path, run):
    """
    Loads multiple pickle files, extracts a specific column, and plots a combined histogram.
    
    Args:
        pickle_pattern (str): Glob pattern for files (e.g., 'data/*.pkl' or '*.pickle').
        run: run number of data.
    """
    FIGURE_DIR = 'figures/pickle/'+f'{run}'
    directory = Path(FIGURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    
    # 1. Find all matching pickle files
    FILE_PATTERN = f"{base_path}/{run}/*.pkl"          
    print(FILE_PATTERN)
    files = glob.glob(FILE_PATTERN)
    

    print(f"Found {len(files)} files. Loading data...")

    all_data = []
    Q1_all_data = []
    PMT_all_data = []
    PMTslot_all_data = []
    gain_all_data = []
    mu_all_data = []
    N_all_data = []
    Q0_all_data = []
    s0_all_data = []
    Q1_all_data = []
    s1_all_data = []
    w_all_data = []
    alpha_all_data = []
    chi_all_data = []

    columns = ['card', 'channel', 'N', 'Q0', 's0', 'Q1', 's1', 'mu', 'w', 'alpha', 'r2', 'chi/NDF', 'slot', 'position']

    # 2. Loop through files and extract the column
    for i, file_path in enumerate(files):
        try:
            # Load the dataframe
            my_list = pd.read_pickle(file_path)
            df =  pd.DataFrame.from_records(my_list[1], columns=columns)
            
            # Extract the column, drop NaNs, and add to list
            # tidy up data
            valid_slots = df['slot'].dropna()
            most_common = valid_slots.mode()
            df['slot'] = df['slot'].fillna(most_common[0])
            
            valid_positions = df['position'].dropna()
            most_common_position = df['position'].mode()
            df['position'] = df['position'].fillna(most_common_position[0])
            
            gain_series = df['Q1'] - df['Q0']
            gain_all_data.append(gain_series)
            mu_series = df['mu']
            mu_all_data.append(mu_series)
            Q1_series = df['Q1']
            Q1_all_data.append(Q1_series)
            PMT_series = ((df['card'].astype(int) - 1) * 19) + df['channel'].astype(int)
            PMT_all_data.append(PMT_series)
            PMTslot_series = ((df['slot'].astype(int)) * 19) + df['position'].astype(int)
            PMTslot_all_data.append(PMTslot_series)
            N_series = df['N']
            N_all_data.append(N_series)
            Q0_series = df['Q0']
            Q0_all_data.append(Q0_series)
            s1_series = df['s1']
            s1_all_data.append(s1_series)
            s0_series = df['s0']
            s0_all_data.append(s0_series)
            w_series = df['w']
            w_all_data.append(w_series)
            alpha_series = df['alpha']
            alpha_all_data.append(alpha_series)
            chi_series = df['chi/NDF']
            chi_all_data.append(chi_series)
            
            # Optional: Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{len(files)} files...")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            print(df)

    # 3. Concatenate all series into one large array
    combined_Q1_data = pd.concat(Q1_all_data, ignore_index=True)
    combined_gain_data = pd.concat(gain_all_data, ignore_index=True)
    combined_PMT_data = pd.concat(PMT_all_data, ignore_index=True)
    combined_PMTslot_data = pd.concat(PMTslot_all_data, ignore_index=True)
    combined_mu_data = pd.concat(mu_all_data, ignore_index=True)
    combined_N_data = pd.concat(N_all_data, ignore_index=True)
    combined_Q0_data = pd.concat(Q0_all_data, ignore_index=True)
    combined_s1_data = pd.concat(s1_all_data, ignore_index=True)
    combined_w_data = pd.concat(w_all_data, ignore_index=True)
    combined_alpha_data = pd.concat(alpha_all_data, ignore_index=True)
    combined_chi_data = pd.concat(chi_all_data, ignore_index=True)
    combined_s0_data = pd.concat(s0_all_data, ignore_index=True)

    # 4. Plot the Histogram

    fig, ax = plt.subplots(1, 1)
    
    bins = np.arange(100,200)
    bin_centres = bins[:-1] + (bins[1] - bins[0]) / 2
    bin_edges = np.concatenate(([bin_centres[0] - (bin_centres[1] - bin_centres[0]) / 2],
                           (bin_centres[:-1] + bin_centres[1:]) / 2,
                           [bin_centres[-1] + (bin_centres[-1] - bin_centres[-2]) / 2]))

    binned_data, _ = np.histogram(combined_gain_data, bins=bin_edges)


    popt, pcov = curve_fit(gaussian, bin_centres, binned_data, p0=[1, 0, 1])
    amplitude, mean, stddev = popt
    y_fit = gaussian(bin_centres, *popt)

    ax.plot(bin_centres, binned_data)
    plt.plot(bin_centres, y_fit, label='Fitted Gaussian', color='red')
    ax.set_title('Gain')
    ax.set_xlabel('Gain')
    ax.set_ylabel('No. of entries')
    fig.savefig(os.path.join(FIGURE_DIR, f"Run_{run}_gain.png"))
    
    fig_gain, ax_gain = plt.subplots(1, 1)
    mean_gain = combined_gain_data.mean()
    ax_gain.scatter(combined_PMT_data, combined_gain_data, color='red', alpha=0.6, s=10)
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
    ax_gain.set_title('Gain vs. PMT')
    ax_gain.set_xlabel('PMT')
    ax_gain.set_ylabel('Gain')
    fig_gain.savefig(os.path.join(FIGURE_DIR, f"Run_{run}_gains.png"))
        
    fig_mu, ax_mu = plt.subplots(1, 1)
    ax_mu.scatter(combined_PMT_data, combined_mu_data, color='red', alpha=0.6, s=10)  
    ax_mu.set_title('mu vs. PMT')
    ax_mu.set_xlabel('PMT')
    ax_mu.set_ylabel('mu')
    fig_mu.savefig(os.path.join(FIGURE_DIR, f"Run_{run}_mu.png"))
    
    fig_gainmu, ax_gainmu = plt.subplots(1, 1)
    ax_gainmu.scatter(combined_gain_data, combined_mu_data, color='red', alpha=0.6, s=10)  
    ax_gainmu.set_title('gain vs. mu')
    ax_gainmu.set_xlabel('gain')
    ax_gainmu.set_ylabel('mu')
    fig_gainmu.savefig(os.path.join(FIGURE_DIR, f"Run_{run}_gainVSmu.png"))
    
    eventDisplay = EventDisplay.EventDisplay() 
    eventDisplay.load_mPMT_positions('mPMT_2D_projection_angles.csv')
    ev_disp_data = eventDisplay.process_data(combined_PMTslot_data,combined_gain_data,sum_data=False, average_data=False)
    eventDisplay.plotEventDisplay(ev_disp_data,color_norm=colors.LogNorm())
    plt.savefig(os.path.join(FIGURE_DIR, f"Run_{run}_Event_Dislay.png"))
        
    fig_N, ax_N = plt.subplots(1, 1)
    ax_N.scatter(combined_PMT_data, combined_N_data, color='red', alpha=0.6, s=10)  
    ax_N.set_title('N vs. PMT')
    ax_N.set_xlabel('PMT')
    ax_N.set_ylabel('N')
    fig_N.savefig(os.path.join(FIGURE_DIR, f"Run_{run}_N.png"))
        
    fig_Q0, ax_Q0 = plt.subplots(1, 1)
    ax_Q0.scatter(combined_PMT_data, combined_Q0_data, color='red', alpha=0.6, s=10)  
    ax_Q0.set_title('Q0 vs. PMT')
    ax_Q0.set_xlabel('PMT')
    ax_Q0.set_ylabel('Q0')
    fig_Q0.savefig(os.path.join(FIGURE_DIR, f"Run_{run}_Q0.png"))
        
    fig_s0, ax_s0 = plt.subplots(1, 1)
    ax_s0.scatter(combined_PMT_data, combined_s0_data, color='red', alpha=0.6, s=10)  
    ax_s0.set_title('s0 vs. PMT')
    ax_s0.set_xlabel('PMT')
    ax_s0.set_ylabel('s0')
    fig_s0.savefig(os.path.join(FIGURE_DIR, f"Run_{run}_s0.png"))
        
    fig_Q1, ax_Q1 = plt.subplots(1, 1)
    ax_Q1.scatter(combined_PMT_data, combined_Q1_data, color='red', alpha=0.6, s=10)  
    ax_Q1.set_title('Q1 vs. PMT')
    ax_Q1.set_xlabel('PMT')
    ax_Q1.set_ylabel('Q1')
    fig_Q1.savefig(os.path.join(FIGURE_DIR, f"Run_{run}_Q1.png"))
        
    fig_s1, ax_s1 = plt.subplots(1, 1)
    ax_s1.scatter(combined_PMT_data, combined_s1_data, color='red', alpha=0.6, s=10)  
    ax_s1.set_title('s1 vs. PMT')
    ax_s1.set_xlabel('PMT')
    ax_s1.set_ylabel('s1')
    fig_s1.savefig(os.path.join(FIGURE_DIR, f"Run_{run}_s1.png"))
        
    fig_w, ax_w = plt.subplots(1, 1)
    ax_w.scatter(combined_PMT_data, combined_w_data, color='red', alpha=0.6, s=10)  
    ax_w.set_title('w vs. PMT')
    ax_w.set_xlabel('PMT')
    ax_w.set_ylabel('w')
    fig_w.savefig(os.path.join(FIGURE_DIR, f"Run_{run}_w.png"))
        
    fig_alpha, ax_alpha = plt.subplots(1, 1)
    ax_alpha.scatter(combined_PMT_data, combined_alpha_data, color='red', alpha=0.6, s=10)  
    ax_alpha.set_title('alpha vs. PMT')
    ax_alpha.set_xlabel('PMT')
    ax_alpha.set_ylabel('alpha')
    fig_alpha.savefig(os.path.join(FIGURE_DIR, f"Run_{run}_alpha.png"))
        
    fig_chi, ax_chi = plt.subplots(1, 1)
    ax_chi.scatter(combined_PMT_data, combined_chi_data, color='red', alpha=0.6, s=10)  
    ax_chi.set_title('chi/NDF vs. PMT')
    ax_chi.set_xlabel('PMT')
    ax_chi.set_ylabel('chi?NDF')
    fig_chi.savefig(os.path.join(FIGURE_DIR, f"Run_{run}_chi.png"))

    figures = {}
    axes = {}

    j = 0
    col_data = [combined_N_data, combined_Q0_data, combined_s0_data, combined_Q1_data, combined_s1_data, combined_mu_data, combined_w_data, combined_alpha_data, combined_chi_data]
    col = ['N', 'Q0', 's0', 'Q1', 's1', 'mu', 'w', 'alpha', 'chi/NDF']
    for i in col:
       figures[i], axes[i] = plt.subplots(1, 1)
    
       series_x = col_data[j]
       series_y = combined_gain_data

       arr_x = np.asarray(series_x).flatten()
       arr_y = np.asarray(series_y).flatten()

       combined_df = pd.DataFrame({
           'x': arr_x,
           'y': arr_y
       })
       combined_df = combined_df.reset_index(drop=True)

       Q1 = combined_df['x'].quantile(0.25)
       Q3 = combined_df['x'].quantile(0.75)
       IQR = Q3 - Q1

       lower_bound = Q1 - 1.5 * IQR
       upper_bound = Q3 + 1.5 * IQR

       mask = (combined_df['x'] >= lower_bound) & (combined_df['x'] <= upper_bound)
       df_clean = combined_df[mask]

       axes[i].scatter(df_clean['x'], df_clean['y'], color='red', alpha=0.6, s=10)
       axes[i].set_title(f'{i} vs. gain')
       axes[i].set_xlabel(f'{i}')
       axes[i].set_ylabel('gain')

       if i == 'chi/NDF': 
           save = 'chi'
           filename = f"Run_{run}_gainVS{save}.png"
           figures[i].savefig(os.path.join(FIGURE_DIR, filename))
       else:
           filename = f"Run_{run}_gainVS{i}.png"
           figures[i].savefig(os.path.join(FIGURE_DIR, filename))
           plt.close(figures[i])
       j += 1

def main(base_path, run):
    
    create_combined_histogram(base_path, run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read pickle files and plot combined calibration info from all PMTs."
    )
    parser.add_argument("--base-path",   type=str, required=True,
                        help="Directory containing input pickle files")
    parser.add_argument("--run",         type=str, required=True,
                        help="Run number")
    
    args = parser.parse_args()
    main(**vars(args))
