#!/usr/bin/env python3

import sys
import numpy as np
import polars as pl
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

SLICES = ["LEFT-LOWER", "LEFT-UPPER", "RIGHT-UPPER", "RIGHT-LOWER"]


def extract_hardware_sensor_metrics(raw_responses_100_2048):
    """
    Executes entirely using basic integer summations.
    Input: A binary block of (100 clusters, 2048 spatial challenges) for 1 snapshot
    """
    # 1. Compute Center Points (x0_proxies) via PopCount
    # Outputs an array of 100 integers
    x0_proxies = np.sum(raw_responses_100_2048, axis=1)

    # Count how many clusters are completely saturated/unusable for key generation
    # clipped_clusters = np.sum((x0_proxies == 0) | (x0_proxies == 2048))
    # print(clipped_clusters)
    counts, bins = np.histogram(x0_proxies, bins=np.logspace(5, 11, 20, base=2))
    # print(f'x0_proxies counts: {counts}')
    # print(f'x0_proxies bins: {bins}')

    # 2. Extract Global Environmental Bias Feature (Mean Shift)
    # This value tracks linearly with global Temperature / Voltage droop
    global_bias = np.sum(x0_proxies) // 100

    # 2. METASTABILITY JITTER PROXY (k) -> Track Junction Temperature
    # Subtract adjacent elements WITHIN THE SAME ROW (axis=1)
    # This counts how noisy/metastable each individual cluster's transition is
    inner_row_transitions = np.abs(
        raw_responses_100_2048[:, 1:] - raw_responses_100_2048[:, :-1]
    )

    # Sum the bitwise transitions per cluster, then average across all 100 clusters
    global_temperature_proxy = np.sum(inner_row_transitions) // 100
    # 3. Extract the Slope Proxy (k_proxy) without floating-point curve fitting
    # We find the absolute spatial derivative across adjacent clusters
    # High derivative = Sharp Sigmoid (Low Temp) | Low derivative = Flat Sigmoid (High Temp)
    spatial_derivatives = np.abs(x0_proxies[1:] - x0_proxies[:-1])
    global_slope_proxy = np.sum(spatial_derivatives) // 99

    return global_temperature_proxy, global_bias, global_slope_proxy


datasets = {
    # Out of the freezer, the board sensor read ~2C but it quickly rose up to 37 C
    0: "data/ergodicity_cluster_temp/crps_CMOD_S7-210376B5B7B5_temp_0.csv",
    # At room temperature ~ 41 C
    41: "data/ergodicity_cluster/crps_CMOD_S7-210376B5B7B5.csv",
    90: "data/ergodicity_cluster_temp/crps_CMOD_S7-210376B5B7B5_temp_90.csv",
    # With 180 C heatgun so 150 C internal
    150: "data/ergodicity_cluster_temp/crps_CMOD_S7-210376B5B7B5_temp_150.csv",
}

results = []

for temp, csv in datasets.items():
    df = pl.read_csv(csv)
    for slice_loc in SLICES:
        # if slice_loc != 'LEFT-LOWER':
        #     continue
        responses = (
            df.filter(pl.col("slice") == slice_loc)["response"]
            .to_numpy()
            .reshape((100, 2**11, 21))
        )
        global_temp_proxy, global_bias, global_slope_proxy = (
            extract_hardware_sensor_metrics(responses)
        )
        results.append(
            {
                "temperature": temp,
                "slice": slice_loc,
                "global_temp_proxy": global_temp_proxy,
                "global_bias": global_bias,
                "global_slope_proxy": global_slope_proxy,
            }
        )

        # # Count spatial transitions within the SAME cluster
        # # High count = high thermal noise (High Temperature)
        # # Low count = sharp transition (Low Temperature)
        # cluster_jitter = np.sum(np.abs(Y[1:] - Y[:-1]))
        # global_thermal_sensor = np.sum(cluster_jitter_across_all_100_clusters)

df = pl.DataFrame(results).sort(['slice', 'temperature'])
with pl.Config(tbl_rows=-1):
    print(df)

df.write_csv(sys.stdout)

# crps = []

# for csv in csvs:
#     df = pl.read_csv(csv)

#     for slice in SLICES:
#         print(slice)
#         responses = df.filter(pl.col('slice') == slice)['response'].to_numpy().reshape((100, 2**11, 21))

#         # # 1. Compute the raw proxy without sorting across the temporal samples
#         # # Shape remains (100, 21) where 100 is clusters and 21 is time samples
#         x0_proxy = responses.sum(axis=1)

#         # 2. Linear Adjacent Pairing (No Autocorrelation)
#         # # Pair 0 vs 1, 2 vs 3, 4 vs 5 ... up to 98 vs 99.
#         # # This yields exactly 50 completely independent key bits from 100 clusters.
#         deltas = x0_proxy[0::2, :] - x0_proxy[1::2, :] # Shape: (50, 21)


#         # 2. Low pass digital Filter
#         # deltas = []
#         # ws = 1
#         # for i in range(0+ws, 100-ws):
#         #     avg = np.sum(x0_proxy[[i-1, i+1], :], axis=0) / 2
#         #     deltas.append(x0_proxy[i,:] - avg)
#         # deltas = np.array(deltas)

#         # 3. Compute Key Metrics cleanly
#         binary_key_matrix = np.where(deltas >= 0, 1, 0)
#         ref_key = np.where(np.mean(binary_key_matrix, axis=1) >= 0.5, 1, 0)

#         print(f'Uniformity: {ref_key.mean()}')

#         # Calculate reliability across real chronological time
#         reliability = np.array([(binary_key_matrix.T[i] == ref_key).mean() for i in range(21)])
#         print(f'Reliability: {reliability}')
#         # if slice == 'RIGHT-UPPER':
#         crps.append(ref_key)


# crps = np.array(crps)
# uniqueness = pdist(crps, 'hamming')
# print(squareform(uniqueness))
