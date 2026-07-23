# Laser ball Analysis

## Waveform processing for charge analysis

These tools are for processing WCTE hardware trigger PMT waveforms from WCTE offline ROOT files into per-hit timing and charge outputs, collated per PMT rather than per event. This is useful for charge distributions that are per PMT, but less useful for timing analysis that needs relative times across different channels in the same event.

It consists of `process_pmt_waveforms.py` for the main processing pipeline, and optionally `find_pmt_time_offsets.py` for computing PMT timing offsets relative to the monitor PMT.

## Dependencies

Requires `numpy`, `awkward`, `uproot`, and the `analysis_tools` package.

See the `analysis_tools` GitHub repository for details and full documentation: `https://github.com/WCTE/analysis_tools`

Typical minimal setup:

```bash
git clone git@github.com:WCTE/analysis_tools.git
python -m pip install -e ./analysis_tools
```

## `process_pmt_waveforms.py`

Reads raw waveform data, applies pulse finding, CFD timing, and charge integration, and writes a ROOT file with one tree per mPMT card containing hit times, charges, PMT hardware channel IDs / PMT geometry position IDs, and event numbers, collated over the entire run or part file.

Files and events are processed in batches to manage memory usage, so that an entire run can be processed in a single execution. The processing is optimised so that a typical full run takes less than one hour on a single CPU core.

ROOT's `hadd` utility can be used to merge together the output files from multiple parts into a single file per run, if desired, allowing processing to be parallelised over part files.

### Usage

```bash
python process_waveforms_for_charge_distributions.py \
    --base-path /path/to/data/ \
    --run 2307 \
    --out-file output/processed_hits.root \
    [--part 0] \
    [--batch-size "100 MB"] \
    [--max-events 1000] \
    [--offset-file offsets/pmt_offsets.root] \
    [--no-monitor-pmt] \
    [--no-pulse-finding]
```

### Options

| Option                 | Required    | Description                                                                        |
|------------------------|-------------|------------------------------------------------------------------------------------|
| `--base-path`          | yes         | Directory containing input ROOT files                                              |
| `--run`                | yes         | Run number                                                                         |
| `--out-file`           | yes         | Output ROOT file path                                                              |
| `--part`               | no          | Part number to read; omit to read all parts                                        |
| `--batch-size`         | no          | Size of each batch as number of events or size in MB (default: `100 MB`)           |
| `--max-events`         | no          | Cap on total events to process; useful for quick debug runs                        |
| `--offset-file`        | conditional | Required when monitor mode is enabled; invalid when monitor mode is disabled       |
| `--[no-]monitor-pmt`   | no          | Enable/disable monitor PMT to deterimine expected hit locations (default: enabled) |
| `--[no-]pulse-finding` | no          | Enable/disable pulse-finding (default: enabled)                                    |

### Charge integration window modes

Three methods are available for defining the charge integration window, controlled by `--monitor-pmt` / `--no-monitor-pmt` and `--pulse-finding` / `--no-pulse-finding`. See the `find_pmt_time_offsets.py` section for how to create the offset file required for monitor-PMT modes.

In all cases, the charge integration follows the standard algorithm from `analysis_tools.waveform_processing`, which simply sums 7 or 8 samples' charge values from 5 samples before the peak sample to 1 sample after, plus the 2nd sample after the peak position if that sample is positive.

The three modes differ in how the peak position is defined:

- **Hybrid combined method** _(default)_: `--monitor-pmt --pulse-finding` (both defaults). Requires `--offset-file`. The peak position is determined from a pulse found within +/- 5 samples of the expected location offset from the monitor PMT pulse; if no pulse is found, the expected location itself is used as fallback to retain the pedestal and below-threshold pulses.
- **Fixed offset from monitor PMT**: `--monitor-pmt --no-pulse-finding`. Requires `--offset-file`. The pulse position is assumed to be located directly offset from the monitor PMT pulse.
- **Pulse-finding only** (no monitor PMT): `--no-monitor-pmt --pulse-finding`. No `--offset-file` should be given. The peak position of the first pulse found in the entire waveform is used; if no pulse is found, the maximum-amplitude sample is used as fallback to retain the pedestal and below-threshold pulses.

`--no-monitor-pmt` and `--no-pulse-finding` are mutually exclusive and cannot be used together.

**Monitor PMT** (`--monitor-pmt`) pulses are used to determine the expected location of each PMT's hit, using offset values in the file provided by `--offset-file`. The monitor PMT (card 131, channel 16) pulse is found per event, and each PMT's expected hit location is determined by the monitor PMT pulse plus the stored offset for the given PMT. Events with zero or more than one monitor PMT pulse are skipped, and channels with missing offsets are skipped. The expected pulse location is either used for a local search region to perform pulse finding (default) or directly defines the assumed pulse position for charge integration (`--no-pulse-finding`). See the `find_pmt_time_offsets.py` documentation for how to create the offset file required for these modes.

**Pulse finding** (`--pulse-finding`) runs the standard pulse finding algorithm from WCTE `analysis_tools.pulse_finding`. The search region for pulse finding is either defined to locate pulses with peaks +/-5 sample around the expected location from the monitor PMT (default) or in the entire waveform (`--no-monitor-pmt`).

### Input files

Expected filename pattern: `WCTE_offline_R{run}S0P{part}.root` under `--base-path`, containing tree `WCTEReadoutWindows` in the standard WCTE offline data ROOT format.

### Output format

One ROOT tree per mPMT card, named `card_{id}_slot_{slot}` (or `card_{id}` for cards 13x that still have their waveforms processed, despite not being actual mPMTs and so have no slot number), each with one entry per hit and the following branches:

| Branch           | Type  | Description                                                                |
|------------------|-------|----------------------------------------------------------------------------|
| `channel_ids`    | int   | PMT channel ID within the card                                             |
| `pmt_positions`  | int   | PMT position from detector mapping                                         |
| `event_numbers`  | int   | Source event number                                                        |
| `times`          | float | Hit time in ns, absolute within the readout window                         |
| `charges`        | float | Integrated charge in ADC counts                                            |
| `is_pulse_found` | bool  | For `--pulse-finding`, whether a pulse was found or below threshold charge |

---

## `find_pmt_time_offsets.py`

Computes the median coarse-count timing offset of pulses found at each PMT relative to pulses found from the monitor PMT (card 131, channel 16). Run this first if you intend to use monitor-PMT modes in `process_pmt_waveforms.py`, to get all PMTs' typical hit timing offsets from the monitor PMT.

Files and events are processed in batches to manage memory usage, so that an entire run can be processed in a single execution. The processing is optimised so that a typical full run takes less than one hour on a single CPU core.

A single offset value is produced per PMT determined using all events, so if running over individual part files it does not make sense to use `hadd` (or any other method) to combine them together into one file. In the case of running over individual part files, each output file would be used for processing the corresponding part file. But unless PMT timing offsets are expected to change within a run, the offsets should be the same for every part file, and so each part's offset file would be identical.

For runs where the laserball position is moved within the run, it is important to produce and use individual offset files for each part file, since the offsets will change as the laserball position changes.

### Usage

```bash
python find_pmt_time_offsets.py \
    --base-path /path/to/data/ \
    --run 2307 \
    --out-file offsets/pmt_offsets.root \
    [--part 0] \
    [--batch-size "100 MB"] \
    [--max-events 1000]
```

### Options

| Option         | Required | Description                                                              |
|----------------|----------|--------------------------------------------------------------------------|
| `--base-path`  | yes      | Directory containing input ROOT files                                    |
| `--run`        | yes      | Run number                                                               |
| `--out-file`   | yes      | Output ROOT file path for the offsets                                    |
| `--part`       | no       | Part number to read; omit to read all parts                              |
| `--batch-size` | no       | Size of each batch as number of events or size in MB (default: `100 MB`) |
| `--max-events` | no       | Cap on total events to process                                           |

### Input files

Expected filename pattern: `WCTE_offline_R{run}S0P{part}.root` under `--base-path`, containing tree `WCTEReadoutWindows` in the standard WCTE offline data ROOT format.

### Output format

A single ROOT tree `pmt_offsets` with one entry per PMT:

| Branch           | Type | Description                                                             |
|------------------|------|-------------------------------------------------------------------------|
| `mpmt_card_id`   | int  | mPMT card ID                                                            |
| `pmt_channel_id` | int  | Channel ID within the card                                              |
| `offset`         | int  | Median coarse-count offset of PMT hits relative to the monitor PMT hits |

---

## Typical workflow

```bash
# Step 1: compute offsets
python find_pmt_time_offsets.py \
    --base-path R2307/data --run 2307 \
    --out-file R2307/offsets/pmt_offsets.root

# Step 2 (default): process waveforms with hybrid monitor+offset mode
python process_waveforms_for_charge_distributions.py \
    --base-path R2307/data --run 2307 \
    --out-file R2307/processed/hits_hybrid.root \
    --offset-file R2307/offsets/pmt_offsets.root

# Step 2 alternative: fixed-offset mode (no pulse finding)
python process_waveforms_for_charge_distributions.py \
    --base-path R2307/data --run 2307 \
    --out-file R2307/processed/hits_fixed_offset.root \
    --offset-file R2307/offsets/pmt_offsets.root \
    --no-pulse-finding

# Step 2 alternative: pulse-finding-only mode (no monitor PMT, and step 1 can be skipped)
python process_waveforms_for_charge_distributions.py \
    --base-path R2307/data --run 2307 \
    --out-file R2307/processed/hits_no_monitor.root \
    --no-use-monitor-pmt

# Step 3: run the fitting code
python bellamy_fitting.py --base-path R2307/processed/hits_hybrid.root --run 2307 --card 1 --out-file ./results/
```

## Charge distribution analysis

An example notebook for plotting histograms of charge distributions per PMT is `lb_charge_distributions.ipynb`. This notebook reads output ROOT files from `process_pmt_waveforms.py` and produces charge distribution histograms for each PMT, comparing the three different approaches for files created for run 2307 using each approach.


## PMT Charge Distribution Fitting Script

Description

This script fits PMT charge distributions using the Bellamy model with the lmfit library. It takes as input the output of the waveform processinig code as described above. The charging fitting for a single mPMT can be done in a few minutes and for the all PMTs ~2 hours. 

Features

Fitting Models:
Bellamy Model: Fits PMT response using a combination of Gaussian and pedestal response functions.
Gaussian Model: For the pedestal peak

Additional Requirements:
EventDisplay (WCTE event display code, included in this branch of the analysis code).

Usage

Run the script from the command line with the following arguments:

python bellamy_fitting.py --base-path <input_directory> --run <run_number> --card <card_number> --out-file <output_path>

--base-path: Directory containing input ROOT files.
--run: Run number (e.g., 2307).
--card: Card number (identifier for the PMT card).
--out-file: Path to save the output.

Example

python bellamy_fitting.py --base-path /path/to/output --run 2307 --card 1 --out-file ./results/  # reads /path/to/output/2307/processed_hits.root

summary plots can then be generated with the pickle_to_plot.py script


Contact

For issues or questions related to the fitting code, email jnugent@ic.ac.uk.

