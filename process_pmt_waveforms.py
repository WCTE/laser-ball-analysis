#!/usr/bin/env python3
"""
Combined PMT waveform processing pipeline.

Reads raw waveform data from ROOT file(s), applies baseline subtraction and
pulse finding, then writes a new ROOT file with one entry per (card, PMT),
storing vectors of hit times and integrated charges.

By default, monitor-PMT offsets are used with hybrid pulse finding.
Use --no-pulse-finding to use fixed offset windows, or --no-monitor-pmt
to disable monitor-based modes and use pulse-finding only.

Usage:
    python process_pmt_waveforms.py \
        --base-path /path/to/root/files \
        --run 1234 \
        --out-file output.root \
        [--part 0] \
        [--batch-size "100 MB"] \
        [--max-events 1000] \
        [--offset-file offsets.root] \
        [--no-monitor-pmt] \
        [--no-pulse-finding]
"""

import os
import argparse

import numpy as np
import awkward as ak
import uproot

from analysis_tools.pulse_finding import do_pulse_finding_fast
from analysis_tools.waveform_processing import WaveformProcessingmPMT
from analysis_tools.wcte_pmt_mapping import PMTMapping

_wf_processor = WaveformProcessingmPMT()
_mapping = PMTMapping()

BRANCHES = [
    "event_number",
    "window_time",
    "pmt_waveforms",
    "pmt_waveform_times",
    "pmt_waveform_mpmt_card_ids",
    "pmt_waveform_pmt_channel_ids",
]
TREE = "WCTEReadoutWindows"


def process_batch(batch, offsets=None, hybrid_offset_pulse=False):
    """
    Flatten one iterate batch, run vectorised pulse finding, CFD timing and
    charge integration, and return the results grouped by card.

    Mirrors the logic of do_hit_processing() in hw_trigger_wf_processing.py.
    If offsets are provided, these are used along with the monitor PMT hit time to set the time window, otherwise pulse
    finding is used. In hybrid mode, offsets define an initial per-waveform search region, and that region is chosen
    so any pulse returned by pulse-finding is automatically within +/-3 samples of the offset-seeded peak. If no
    pulse is found, the direct offset index is used.
    For each waveform, at least one pulse is always returned: if pulse-finding finds nothing, the sample with the
    highest amplitude is used as the fallback peak position.

    Parameters
    ----------
    batch : ak.Array of events
    offsets : str root file of PMT to monitor PMT coarse-count offsets (optional, default None)
    hybrid_offset_pulse : bool
        If True and offsets are provided, use offset-defined windows as seeds and pulse-find within each seed window.

    Returns
    -------
    dict  {card_id: (hit_times, hit_charges)}
          hit_times in ns (absolute, relative to start of readout window)
          hit_charges in ADC counts (integrated over the pulse)
    """
    slice_len    = 12
    peak_position = 8
    time_base_ns  = 8.0
    hybrid_window_tolerance = 5

    # loop over events, process one event at a time to ease matching to monitor PMT
    hit_card_ids = []
    hit_channel_ids = []
    hit_times = []
    hit_charges = []
    hit_event_nums = []
    warn_no_hit = True
    warn_multiple_hit = True
    for event in batch:
        waveforms = ak.to_numpy(event["pmt_waveforms"])
        wf_times = ak.to_numpy(event["pmt_waveform_times"] + event["window_time"])
        card_ids = ak.to_numpy(event["pmt_waveform_mpmt_card_ids"])
        channel_ids = ak.to_numpy(event["pmt_waveform_pmt_channel_ids"])
        event_num = ak.to_numpy(event["event_number"])

        # Vectorised pulse finding

        n_waveforms, waveform_length = waveforms.shape
        min_peak_sample = peak_position
        max_peak_sample = waveform_length - (slice_len - peak_position)
        if offsets is not None:
            is_mon_pmt = (card_ids == 131) & (channel_ids == 16)
            mon_hit_wf_index, mon_hit_indices = do_pulse_finding_fast(waveforms[is_mon_pmt])
            if len(mon_hit_wf_index) < 1:
                if warn_no_hit:
                    print("[WARN] Event with no monitor PMT hit found... skipping all events like this")
                    warn_no_hit = False
                continue
            if len(mon_hit_wf_index) > 1:
                if warn_multiple_hit:
                    print("[WARN] Event with multiple monitor PMT hits found... skipping all events like this")
                    warn_multiple_hit = False
                continue
            mon_hit_index = mon_hit_indices[0]
            hit_wf_time_offsets = (wf_times[is_mon_pmt][0] - wf_times) // 8
            hit_offsets = offsets[card_ids, channel_ids]
            hit_indices = np.clip(
                mon_hit_index + hit_offsets + hit_wf_time_offsets,
                min_peak_sample, max_peak_sample,
            ).astype(np.intp)
            good = hit_offsets != -999
            hit_wf_index = np.arange(n_waveforms)[good]
            hit_indices = hit_indices[good]

            if hybrid_offset_pulse and len(hit_wf_index) > 0:
                # Pick search bounds so do_pulse_finding_fast valid columns map to offset +/-3
                # With search_start = offset-(3+4) and len = (3+4)+(3+2)+1 = 13, valid cols [4..10] map to
                # offset-5 to offset+5
                search_pre = hybrid_window_tolerance + 4
                search_post = hybrid_window_tolerance + 2
                search_len = search_pre + search_post + 1

                search_start = np.clip(hit_indices - search_pre, 0, waveform_length - search_len).astype(np.intp)
                search_slice_idx = search_start[:, None] + np.arange(search_len, dtype=np.intp)
                search_hit_idx = np.arange(len(hit_wf_index), dtype=np.intp)[:, None]
                search_samples = waveforms[hit_wf_index][search_hit_idx, search_slice_idx]

                seed_rows, seed_cols = do_pulse_finding_fast(search_samples)
                if len(seed_rows) > 0:
                    hit_indices[seed_rows] = search_start[seed_rows] + seed_cols
                    np.clip(hit_indices, min_peak_sample, max_peak_sample, out=hit_indices)
        else:
            hit_wf_index, hit_indices = do_pulse_finding_fast(waveforms)
            good = (hit_indices >= min_peak_sample) & (hit_indices <= max_peak_sample)
            hit_wf_index = hit_wf_index[good]
            hit_indices = hit_indices[good]
            # Fallback: for waveforms with no found pulse use argmax of the waveform.
            no_hit_rows = np.full(n_waveforms, True, dtype=bool)
            no_hit_rows[hit_wf_index] = False
            fallbacks = np.clip(
                np.argmax(waveforms[no_hit_rows], axis=1),
                min_peak_sample, max_peak_sample,
            ).astype(np.intp)
            hit_wf_index = np.concatenate([hit_wf_index, np.where(no_hit_rows)[0]])
            hit_indices = np.concatenate([hit_indices, fallbacks])
            order = np.argsort(hit_wf_index, kind="stable")
            hit_wf_index = hit_wf_index[order]
            hit_indices = hit_indices[order]

        # build per-hit waveform slices (mirrors hw_trigger_wf_processing.py)

        start_sample = hit_indices - peak_position
        local_idx = np.arange(slice_len)
        slice_idx = start_sample[:, None] + local_idx[None, :]  # (n_hits, slice_len)
        hit_idx = np.arange(len(hit_indices))[:, None]
        waveform_samples = waveforms[hit_wf_index][hit_idx, slice_idx]  # (n_hits, slice_len)

        # vectorised CFD time and charge
        _, cfd_time_corr, _, _ = _wf_processor.cfd_vectorized(waveform_samples)
        # time in ns relative to start of readout window
        hit_times.extend(wf_times[hit_wf_index] + (start_sample + cfd_time_corr) * time_base_ns)
        hit_charges.extend(_wf_processor.charge_vectorized_mPMT_method(waveform_samples, peak_position))
        hit_card_ids.extend(card_ids[hit_wf_index])
        hit_channel_ids.extend(channel_ids[hit_wf_index])
        hit_event_nums.append(np.repeat(event_num, len(hit_wf_index)))

    if len(hit_times) == 0:
        print("[WARN] No valid hits (probably because no monitor PMT pulses to offset to)")
        return {}

    hit_times = np.array(hit_times)
    hit_charges = np.array(hit_charges)
    hit_card_ids = np.array(hit_card_ids)
    hit_channel_ids = np.array(hit_channel_ids)
    hit_event_nums = np.concatenate(hit_event_nums)

    # Group results by card
    unique_cards, hit_card = np.unique(hit_card_ids, axis=0, return_inverse=True)

    # Sort hits by card, then slice contiguous blocks
    sort_order = np.argsort(hit_card, kind="stable")
    hit_card_sorted = hit_card[sort_order]
    hit_channels_sorted = hit_channel_ids[sort_order]
    hit_times_sorted = hit_times[sort_order]
    hit_charges_sorted = hit_charges[sort_order]
    event_num_sorted = hit_event_nums[sort_order]
    split_points = np.searchsorted(hit_card_sorted, np.arange(len(unique_cards) + 1))
    hits_by_card = {}
    for i, card in enumerate(unique_cards):
        start, end = split_points[i], split_points[i + 1]
        hits_by_card[card] = {"channel_ids": hit_channels_sorted[start:end],
                              "times": hit_times_sorted[start:end],
                              "charges": hit_charges_sorted[start:end],
                              "event_numbers": event_num_sorted[start:end]}
    return hits_by_card


def run_pipeline(run, base_path, out_file, part=None, batch_size="100 MB", max_events=None,
                 offset_file=None, monitor_pmt=True, pulse_finding=True):

    hybrid_offset_pulse = monitor_pmt and pulse_finding
    mode = "hybrid" if hybrid_offset_pulse else ("fixed-offset" if monitor_pmt else "pulse-finding-only")
    print(f"[INFO] Processing mode: {mode}")

    offsets = None
    if monitor_pmt:
        offset_arrays = uproot.open(offset_file)["pmt_offsets"].arrays()
        cards = offset_arrays["mpmt_card_id"]
        channels = offset_arrays["pmt_channel_id"]
        offset_values = offset_arrays["offset"]
        offsets = np.full((np.max(cards)+1, np.max(channels)+1), -999)
        offsets[cards, channels] = offset_values

    part_str = part if part is not None else "*"
    file_spec = os.path.join(base_path, f"WCTE_offline_R{run}S0P{part_str}.root:{TREE}")
    print(f"[INFO] Reading from {file_spec}")

    n_total = sum((uproot.open(f"{f}:{o}").num_entries for f, o in uproot._util.regularize_files(file_spec, False)))
    if max_events is not None:
        n_total = min(n_total, max_events)
    print(f"[INFO] {n_total} events to process")

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

    with uproot.recreate(out_file) as root_out:
        iterate_kwargs = {"expressions": BRANCHES,
                          "step_size": int(batch_size) if str(batch_size).isdigit() else batch_size,
                          "library": "ak"}
        n_events = 0
        n_hits = 0
        for batch in uproot.iterate(file_spec, **iterate_kwargs):
            if max_events is not None:
                remaining = max_events - n_events
                if len(batch) > remaining:
                    batch = batch[:remaining]
            n_events += len(batch)
            for card, hits in process_batch(batch, offsets, hybrid_offset_pulse=hybrid_offset_pulse).items():
                n_hits += len(hits["times"])
                slot, positions = _mapping.get_slot_pmt_pos_from_card_pmt_chan(card, hits["channel_ids"])
                hits["pmt_positions"] = positions
                tree = f"card_{card}"
                if slot != -1:
                    tree += f"_slot_{slot}"
                if tree not in root_out:
                    n_baskets = (n_total // n_events) + 2
                    root_out.mktree(tree, hits, initial_basket_capacity=n_baskets, resize_factor=(1+1/n_baskets))
                else:
                    root_out[tree].extend(hits)
            print(f"[INFO]   ... {n_events} events processed with {n_hits} total hits")
            if n_events >= n_total:
                break
    if n_events != n_total:
        print(f"[WARN] Did not process expected number of events ({n_events} of {n_total})")
    print(f"[DONE] Processed {n_events} events with {n_hits} total hits --> {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PMT waveforms from ROOT files and write hits to a new ROOT file."
    )
    parser.add_argument("--base-path",   type=str, required=True,
                        help="Directory containing input ROOT files")
    parser.add_argument("--run",         type=int, required=True,
                        help="Run number")
    parser.add_argument("--out-file",    type=str, required=True,
                        help="Output ROOT file path")
    parser.add_argument("--part",        type=int, default=None,
                        help="Part number (P0, P1, ...); omit to read all parts")
    parser.add_argument("--batch-size",  default="100 MB",
                        help="Events per iterate batch as event count or memory limit "
                             "e.g. 500 or '100 MB' (default: '100 MB')")
    parser.add_argument("--max-events",  type=int, default=None,
                        help="Cap on total events to read (optional)")
    parser.add_argument("--offset-file",     type=str, default=None,
                        help="File with PMT offsets to monitor PMT for time window; required unless --no-monitor-pmt")
    parser.add_argument("--monitor-pmt", action=argparse.BooleanOptionalAction, default=True,
                        help="Use monitor PMT and offsets (default: True). Use --no-monitor-pmt to run pulse-finding only")
    parser.add_argument("--pulse-finding", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable pulse finding (default: True). Use --no-pulse-finding for fixed-offset mode")

    args = parser.parse_args()

    if (not args.monitor_pmt) and (not args.pulse_finding):
        parser.error("--no-monitor-pmt and --no-pulse-finding cannot be used together")

    if args.monitor_pmt and args.offset_file is None:
        parser.error("--offset-file is required unless --no-monitor-pmt is set")

    if (not args.monitor_pmt) and args.offset_file is not None:
        parser.error("--offset-file should not be used with --no-monitor-pmt")

    run_pipeline(**vars(args))

