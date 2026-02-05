import os
import random

import numpy as np


def load_power_trace_data(fp_path, data_file_name, sample_length=2048):
    dir = os.path.join(fp_path, data_file_name)
    files = os.listdir(dir)
    data = []

    for f in files:
        if f.endswith(".txt"):
            filepath = os.path.join(dir, f)
            pt = []
            with open(filepath, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    pt.append(float(parts[0]))

            pt = np.array(pt)

            if len(pt) < sample_length:
                print(
                    f"File {filepath} has length {len(pt)}, less than {sample_length}"
                )
                return None

            # Process data
            pt = np.array(pt[:sample_length])

            if len(pt) != sample_length:
                print(
                    f"Verification failed: File {filepath} has length {len(pt)}, expected {sample_length}"
                )
                return None

            data.append(pt)

    m = len(data)
    n = len(data[0])
    print(f"Found and Loaded {m} files from {data_file_name}, Dimension: {m} x {n}")

    return data


def load_vindata_mcu_sel(
    fp_path_train,
    fp_path_test,
    selected_train_mcus,
    selected_test_mcus,
    case_num,
    trace_len,
    samples_per_mcu_pristine,
    samples_per_mcu_per_case_compromised,
    random_selection_train,  # <-- Flag to control selection method
    train_ratio=2,
    noise_level=0.0,
    random_selection_test=False,
):
    train_pristine_data = []
    train_compromised_data = []
    test_pristine_data = []
    test_compromised_data = []

    for mcu in selected_train_mcus:
        for case in case_num:
            filename_prefix = f"RAW_MCU{mcu}_PSC1_C{case}"
            all_available_traces = load_power_trace_data(
                fp_path_train, filename_prefix, trace_len
            )

            num_to_select = (
                samples_per_mcu_pristine
                if case == 0
                else samples_per_mcu_per_case_compromised
            )

            k = min(num_to_select, len(all_available_traces))
            if k < num_to_select:
                print(
                    f"Warning: Requested {num_to_select} samples for {filename_prefix}, but only {k} were available."
                )

            if random_selection_train and k > 0:
                selected_traces = random.sample(
                    all_available_traces, (k // train_ratio)
                )
            else:
                selected_traces = all_available_traces[:k]

            if case == 0:
                train_pristine_data.extend(selected_traces)
            else:
                train_compromised_data.extend(selected_traces)

    for mcu in selected_test_mcus:
        for case in case_num:
            filename_prefix = f"RAW_MCU{mcu}_PSC1_C{case}"
            all_available_traces = load_power_trace_data(
                fp_path_test, filename_prefix, trace_len
            )

            if noise_level > 0.0:
                all_available_traces = [
                    trace + np.random.normal(0, noise_level, size=trace.shape)
                    for trace in all_available_traces
                ]

            num_to_select = (
                samples_per_mcu_pristine
                if case == 0
                else samples_per_mcu_per_case_compromised
            )

            k = min(num_to_select, len(all_available_traces))
            if k < num_to_select:
                print(
                    f"Warning: Requested {num_to_select} samples for {filename_prefix}, but only {k} were available."
                )

            if random_selection_test and k > 0:
                selected_traces = random.sample(all_available_traces, k)
            else:
                selected_traces = all_available_traces[:k]

            if case == 0:
                test_pristine_data.extend(selected_traces)
            else:
                test_compromised_data.extend(selected_traces)

    # Truncate all_pristine_data and all_compromised_data to trace_len samples per element
    train_pristine_data = [trace[:trace_len] for trace in train_pristine_data]
    train_compromised_data = [trace[:trace_len] for trace in train_compromised_data]

    test_pristine_data = [trace[:trace_len] for trace in test_pristine_data]
    test_compromised_data = [trace[:trace_len] for trace in test_compromised_data]

    return (
        train_pristine_data,
        train_compromised_data,
        test_pristine_data,
        test_compromised_data,
    )


def compute_prides(data, n):
    prides = []
    for d in data:
        p = np.zeros(n).astype(int)  # len(p) = n

        # Ensure we don't index out of bounds if d is small
        if len(d) <= n:
            limit = len(d) - 1
            if limit > 0:
                delta = d[1 : limit + 1] - d[:limit]
                p[:limit][delta > 0] = 1
        else:
            delta = d[1 : n + 1] - d[:n]  # Length is n
            p[delta > 0] = 1  # p has length n, so this matches perfectly

        prides.append(p)
    return prides
