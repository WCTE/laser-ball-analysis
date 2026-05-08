#!/usr/bin/env python3
"""
Script to determine PMT timing offsets (in coarse counts) relative to laser monitor PMT.

Reads raw waveform data from ROOT file(s), pulse finding to all hits in each event, finds the offset to the monitor PMT
for each hit. For each PMT, the median offset of all hits of that PMT is taken as that PMT's offset.
The offsets are written to a ROOT file with `pmt_offsets` tree with one entry per PMT, with the card ID, channel ID and
median offset value for the PMT.

Usage:
    python find_pmt_time_offsets.py \
        --base-path /path/to/root/files \
        --run 1234 \
        --out-file output.root \
        [--part 0] \
        [--batch-size 100] \
        [--max-events 1000] \
"""

import os
import argparse
from collections import defaultdict
import numpy as np
import awkward as ak
import uproot

from analysis_tools.pulse_finding import do_pulse_finding_fast
from analysis_tools.waveform_processing import WaveformProcessingmPMT

_wf_processor = WaveformProcessingmPMT()

BRANCHES = [
    "pmt_waveforms",
    "pmt_waveform_mpmt_card_ids",
    "pmt_waveform_pmt_channel_ids",
    "pmt_waveform_times",
    "event_number"
]
TREE = "WCTEReadoutWindows"

def process_batch(batch):
    """
    Flatten one iterate batch, run vectorised pulse finding, CFD timing and
    charge integration, and return the results grouped by PMT key.

    Mirrors the logic of do_hit_processing() in hw_trigger_wf_processing.py.
    For each waveform, at least one pulse is always returned: if pulse finding
    finds nothing, the sample with the highest amplitude is used as the fallback
    peak position.

    Parameters
    ----------
    batch : ak.Array of events

    Returns
    -------
    dict  {(card_id, slot_id, channel_id, pos_id): (hit_times, hit_charges)}
          hit_times in ns (absolute, relative to start of readout window)
          hit_charges in ADC counts
    """
    slice_len    = 12
    peak_position = 8

    # loop over events, process one event at a time to ease matching to monitor PMT
    warn_no_hit = True
    warn_multiple_hit = True
    hit_card_ids = []
    hit_channel_ids = []
    hit_offsets = []
    for event in batch:
        wfs = ak.to_numpy(event["pmt_waveforms"])
        cards = ak.to_numpy(event["pmt_waveform_mpmt_card_ids"])
        channels = ak.to_numpy(event["pmt_waveform_pmt_channel_ids"])
        times = ak.to_numpy(event["pmt_waveform_times"])
        n_waveforms, waveform_length = wfs.shape
        min_peak_sample = peak_position
        max_peak_sample = waveform_length - (slice_len - peak_position)
        hit_wf_index, hit_indices = do_pulse_finding_fast(wfs)
        good = (hit_indices >= min_peak_sample) & (hit_indices <= max_peak_sample)
        hit_wf_index = hit_wf_index[good]
        hit_indices = hit_indices[good]
        # only use events with exactly one monitor PMT hit
        is_mon_pmt = (cards[hit_wf_index] == 131) & (channels[hit_wf_index] == 16)
        if np.sum(is_mon_pmt) < 1:
            if warn_no_hit:
                print("[WARN] Event with no monitor PMT hit found... skipping")
                warn_no_hit = False
            continue
        if np.sum(is_mon_pmt) > 1:
            if warn_multiple_hit:
                print("[WARN] Event with multiple monitor PMT hits found... skipping")
                warn_multiple_hit = False
            continue
        mon_hit_index = hit_indices[is_mon_pmt][0]
        hit_wf_times = times[hit_wf_index]
        hit_wf_time_offsets = (hit_wf_times - hit_wf_times[is_mon_pmt][0])//8
        offsets = hit_indices - mon_hit_index + hit_wf_time_offsets
        hit_card_ids.append(cards[hit_wf_index])
        hit_channel_ids.append(channels[hit_wf_index])
        hit_offsets.append(offsets)

    if len(hit_offsets) == 0:
        return {}

    card_ids_flat     = np.concatenate(hit_card_ids)
    channel_ids_flat  = np.concatenate(hit_channel_ids)
    hit_offsets_flat  = np.concatenate(hit_offsets)

    # Group results by PMT key
    pmts = np.stack([card_ids_flat, channel_ids_flat], axis=1)
    unique_pmts, index, hit_pmt = np.unique(pmts, axis=0, return_index=True, return_inverse=True)

    # Sort hits by PMT, then slice contiguous blocks
    sort_order = np.argsort(hit_pmt, kind="stable")
    hit_pmt_sorted = hit_pmt[sort_order]
    hit_offsets_sorted = hit_offsets_flat[sort_order].astype(int)
    split_points = np.searchsorted(hit_pmt_sorted, np.arange(len(unique_pmts) + 1))
    offsets_by_pmt = {}
    for i, key in enumerate(unique_pmts):
        pmt_key = (int(key[0]), int(key[1]))
        start, end = split_points[i], split_points[i + 1]
        offsets_by_pmt[pmt_key] = (hit_offsets_sorted[start:end])
    return offsets_by_pmt


def run_pipeline(run, base_path, out_file, part=None, batch_size="100 MB", max_events=None):

    part_str = part if part is not None else "*"
    file_spec = os.path.join(base_path, f"WCTE_offline_R{run}S0P{part_str}.root:{TREE}")
    print(f"[INFO] Reading from {file_spec}")

    n_total = sum((uproot.open(f"{f}:{o}").num_entries for f, o in uproot._util.regularize_files(file_spec, False)))
    if max_events is not None:
        n_total = min(n_total, max_events)
    print(f"[INFO] {n_total} events to process")

    iterate_kwargs = {"expressions": BRANCHES, "step_size": int(batch_size) if str(batch_size).isdigit() else batch_size, "library": "ak"}

    # Accumulate hits per PMT across all batches.
    all_hit_offsets = defaultdict(lambda: [])   # {pmt_key: ([hit_times], [hit_charges], [event_nums])}

    n_events = 0
    for batch in uproot.iterate(file_spec, **iterate_kwargs):
        if max_events is not None:
            remaining = max_events - n_events
            if len(batch) > remaining:
                batch = batch[:remaining]
        for pmt_key, h in process_batch(batch).items():
            all_hit_offsets[pmt_key].append(h)

        n_events += len(batch)
        print(f"[INFO]   ... {n_events} events processed, {len(all_hit_offsets)} PMTs seen so far")
        if max_events is not None and n_events >= max_events:
            break

    if n_events == 0:
        print("[WARN] No events found - check run number and part.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

    mpmt_card_ids = np.empty(len(all_hit_offsets), dtype=np.int32)
    pmt_channel_ids = np.empty(len(all_hit_offsets), dtype=np.int32)
    median_hit_offsets = np.empty(len(all_hit_offsets), dtype=np.int32)

    total_hits = 0
    for i, ((card, channel), o) in enumerate(all_hit_offsets.items()):
        mpmt_card_ids[i]    = card
        pmt_channel_ids[i]  = channel
        hit_offsets = np.concatenate(o)
        total_hits += len(hit_offsets)
        median_hit_offsets[i] = np.median(hit_offsets)

    print(f"[INFO] Writing {len(mpmt_card_ids)} PMT offset entries to {out_file}...")
    with uproot.recreate(out_file) as root_out:
        root_out.mktree("pmt_offsets", {
            "mpmt_card_id":   mpmt_card_ids,
            "pmt_channel_id": pmt_channel_ids,
            "offset":         median_hit_offsets
        })

    print(f"[DONE] {len(mpmt_card_ids)} PMTs, processed {total_hits} total hits --> {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PMT waveforms from ROOT files, find the offset (in coarse counts) between monitor PMT hits"
                    "and other PMT hits, and write to a new ROOT file."
    )
    parser.add_argument("--base-path",   type=str, required=True,
                        help="Directory containing input ROOT files")
    parser.add_argument("--run",         type=int, required=True,
                        help="Run number")
    parser.add_argument("--out-file",    type=str, required=True,
                        help="Output ROOT file path")
    parser.add_argument("--part",        type=int, default=None,
                        help="Part number (P0, P1, …); omit to read all parts")
    parser.add_argument("--batch-size",  default="100",
                        help="Events per iterate batch as event count or memory limit "
                             "e.g. 500 or '100 MB' (default: '100')")
    parser.add_argument("--max-events",  type=int, default=None,
                        help="Cap on total events to read (optional)")

    args = parser.parse_args()

    run_pipeline(**vars(args))

