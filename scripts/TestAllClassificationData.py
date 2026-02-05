import os
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DT

from PRIDES_ML import *

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16})


benchmark = "SHA"
# benchmark = 'FFT'
# benchmark = 'BM'
# benchmark = 'BC'

print(f"You have selected the {benchmark} benchmark")
start_time = time.time()  # Record the starting time

# TRAIN DATA PATH
# VIN (FP DATA FROM RAW PSC)
vin_path_train = rf"ALL_RAW_FP_{benchmark}"

# TEST DATA PATH
vin_path_test = rf"ALL_RAW_FP_{benchmark}"

text_to_save = rf"OUTPUT_DIR_{benchmark}"
os.makedirs(text_to_save, exist_ok=True)

selected_train_mcus = [1, 2]
selected_test_mcus = [3, 4, 5, 6]

total_mcus = len(selected_train_mcus) + len(selected_test_mcus)

case_num = [0, 1, 2, 3]  # Chosen cases from dataset
total_cases = len(case_num)


# DOWNSAMPLE SETTINGS
# ratio = 1 (1Msps) (ORIGINAL),
# ratio = 2 (500ksps),
# ratio = 4 (250ksps),
# ratio = 8 (125ksps)
# ratio = 16 (62,500sps)
down_sample_ratios = [1, 2, 4, 8, 16]

# Set True to convert to Binary PRIDES, False to keep as Floating Point
prides = False

# selected_sample_length = 6000 # <------------------------------ [SELECT ACCORDINGLY]
if benchmark == "SHA":
    selected_sample_length = 6000
elif benchmark == "FFT":
    selected_sample_length = 6700
elif benchmark == "BM":
    selected_sample_length = 5000
elif benchmark == "BC":
    selected_sample_length = 6100
else:
    print("ERROR: Incorrect Benchmark Selection, Please Verify ...")

samples_per_mcu_pristine = 600
samples_per_mcu_per_case_compromised = 200

train_ratio = 1  # 1/2/4/5/6/8

total_pristine_sample_size = int(total_mcus * samples_per_mcu_pristine)
total_compromised_sample_size = int(
    total_mcus * (total_cases - 1) * samples_per_mcu_per_case_compromised
)

random_pt_or_prides_train_selection = True  # True # False
if train_ratio == 1:
    random_pt_or_prides_train_selection = False  # no randomness for full dataset

noise_level = 0.0

datasets = [
    (
        "CAD_VIN",
        load_vindata_mcu_sel,
        vin_path_train,
        vin_path_test,
        "float",
    )
]

sample_length_list = [selected_sample_length]

metric_names = ["accuracy", "precision", "recall", "f1", "fpr"]
# List of classifier names
classifier_names = [
    "SVC_linear",
    "KNeighborsClassifier",
    "LogisticRegression",
    "RandomForestClassifier",
    "MLPClassifier_3Layer",
]

# number of times to complete train and test loop
ITER = 1

for down_sample_ratio in down_sample_ratios:
    print(
        f'You have selected the "{datasets[0][0]} - {datasets[0][4]}" dataset for the "{benchmark}" benchmark ***\n'
    )
    print(f"Ratio = {down_sample_ratio} (Taking 1 sample every {down_sample_ratio})\n")
    for sample_length_req in sample_length_list:
        performance_metrics = {
            clf_name: {metric: [] for metric in metric_names}
            for clf_name in classifier_names
        }

        for run in range(1, ITER + 1):
            print(f"Iteration: [{run}/{ITER}]")

            for (
                dataset_name,
                load_fn,
                data_path_train,
                data_path_test,
                dataset_type,
            ) in datasets:
                required_raw_len = sample_length_req

                # Load data
                if dataset_type == "float":
                    (
                        train_pristine_data,
                        train_compromised_data,
                        test_pristine_data,
                        test_compromised_data,
                    ) = load_fn(
                        data_path_train,
                        data_path_test,
                        selected_train_mcus,
                        selected_test_mcus,
                        case_num=case_num,
                        trace_len=required_raw_len,
                        samples_per_mcu_pristine=samples_per_mcu_pristine,
                        samples_per_mcu_per_case_compromised=samples_per_mcu_per_case_compromised,
                        random_selection_train=random_pt_or_prides_train_selection,
                        train_ratio=train_ratio,
                        noise_level=noise_level,
                        random_selection_test=False,
                    )

                    if not train_pristine_data:
                        raise ValueError(
                            f"Data loading failed. requested {required_raw_len} points, check file sizes."
                        )

                    if down_sample_ratio > 1:
                        print(f"Downsampling data by factor of {down_sample_ratio}...")
                        train_pristine_data = [
                            x[::down_sample_ratio] for x in train_pristine_data
                        ]
                        train_compromised_data = [
                            x[::down_sample_ratio] for x in train_compromised_data
                        ]
                        test_pristine_data = [
                            x[::down_sample_ratio] for x in test_pristine_data
                        ]
                        test_compromised_data = [
                            x[::down_sample_ratio] for x in test_compromised_data
                        ]

                    if prides:
                        print("Computing PRIDES (Binary) from Floating Point data...")

                        # Target length for PRIDES is usually len - 1 (differential)
                        current_len = len(train_pristine_data[0])
                        target_len = current_len - 1

                        train_pristine_data = compute_prides(
                            train_pristine_data, target_len
                        )
                        train_compromised_data = compute_prides(
                            train_compromised_data, target_len
                        )
                        test_pristine_data = compute_prides(
                            test_pristine_data, target_len
                        )
                        test_compromised_data = compute_prides(
                            test_compromised_data, target_len
                        )

                        # SWITCH DATASET TYPE TO BINARY
                        dataset_type = "binary"
                        dataset_name = "PRIDES"
                    else:
                        print("Keeping data as Floating Point (Raw Power Trace)...")
                        dataset_name = "PTRACE"

            # Log file
            log_file = os.path.join(
                text_to_save,
                f"{benchmark}_{dataset_name}_TrainRatio{train_ratio}_DS{down_sample_ratio}_Test.txt",
            )

            label_encoder = LabelEncoder()

            # Validate sample lengths
            max_train_sample_size = len(train_pristine_data)
            max_test_sample_size = len(test_pristine_data)

            # Mixing all data
            all_pristine_data = train_pristine_data + test_pristine_data
            all_compromised_data = train_compromised_data + test_compromised_data

            # Validate trace lengths of each sample
            max_trace_len = len(train_pristine_data[0])

            sample_length = max_trace_len

            if any(
                len(x) != max_trace_len
                for x in train_pristine_data
                + train_compromised_data
                + test_pristine_data
                + test_compromised_data
            ):
                print(
                    f"Warning: Trace length mismatch detected. Standardizing to {max_trace_len}..."
                )
                train_pristine_data = [x[:max_trace_len] for x in train_pristine_data]
                train_compromised_data = [
                    x[:max_trace_len] for x in train_compromised_data
                ]
                test_pristine_data = [x[:max_trace_len] for x in test_pristine_data]
                test_compromised_data = [
                    x[:max_trace_len] for x in test_compromised_data
                ]

            # --------------------------------------------------------------------------------------------------------------------------------------
            # Data Summary
            data_summary_text = f"""\n\
    # No. of Total Pristine VOUT/BIN/ILOAD (C0)             = {len(all_pristine_data)}
    # No. of Total Compromised VOUT/BIN/ILOAD (C1–3)        = {len(all_compromised_data)}
    # Length of Each Prides/Trace in VOUT/BIN/ILOAD file    = {max_trace_len} (Effective)
    # Downsample Ratio                                      = {down_sample_ratio}
    """
            print(data_summary_text)

            # Report dimensions
            m_train = max_train_sample_size
            m_test = max_test_sample_size
            n = max_trace_len
            print(f"Train Data Dimensions: {m_train} x {n}")
            print(f"Test  Data Dimensions: {m_test} x {n}")

            # ------------------------------------------------------------------------------------------------------------------------------------------
            with open(log_file, "a") as f:
                if run == 1:
                    f.write(data_summary_text)


            current_len = max_trace_len

            if sample_length == current_len:
                sp = 0
            else:
                sp = np.random.randint(0, current_len - sample_length)
            ep = sp + sample_length

            train_pristine = [x[sp:ep] for x in train_pristine_data]
            train_compromised = [x[sp:ep] for x in train_compromised_data]
            test_pristine = [x[sp:ep] for x in test_pristine_data]
            test_compromised = [x[sp:ep] for x in test_compromised_data]

            # Validate segment lengths
            if any(
                len(x) != sample_length
                for x in train_pristine
                + train_compromised
                + test_pristine
                + test_compromised
            ):
                raise ValueError(
                    f"Segment length mismatch at sl={sample_length}, sp={sp}"
                )

            ## Create feature and label sets
            train_features = np.vstack(train_pristine + train_compromised)
            test_features = np.vstack(test_pristine + test_compromised)


            train_labels = ["pristine"] * len(train_pristine) + ["compromised"] * len(
                train_compromised
            )
            test_labels = ["pristine"] * len(test_pristine) + ["compromised"] * len(
                test_compromised
            )

            # Encode labels
            train_labels_encoded = label_encoder.fit_transform(train_labels)
            test_labels_encoded = label_encoder.transform(test_labels)

            # ------------------------------------------------------------------------------------------------------------------------------------------
            cosine_sim = np.mean(
                [
                    np.dot(train_features[i], test_features[i])
                    / (
                        np.linalg.norm(train_features[i])
                        * np.linalg.norm(test_features[i])
                        + 1e-10
                    )
                    for i in range(min(len(train_features), len(test_features)))
                ]
            )

            # ------------------------------------------------------------------------------------------------------------------------------------------
            train_test_log = f"""
    =============    Train & Test Dataset Info    =============
    Train #Pristine         : {len(train_pristine)}
    Train #Compromised      : {len(train_compromised)}
    Train #Features         : {len(train_features)}, {train_features.shape[1]} features

    Test  #Pristine         : {len(test_pristine)}
    Test  #Compromised      : {len(test_compromised)}
    Test  #Features         : {len(test_features)}, {test_features.shape[1]} features
    ===========================================================
    """
            print(train_test_log)

            # Append to the log file
            with open(log_file, "a") as f:
                if run == 1:
                    f.write(train_test_log)  # Only once per RUN

            if dataset_type == "float":
                print("Scaling Float Data...")
                train_features = np.array(
                    [
                        (feature - np.mean(feature)) / np.std(feature)
                        for feature in train_features
                    ]
                )
                test_features = np.array(
                    [
                        (feature - np.mean(feature)) / np.std(feature)
                        for feature in test_features
                    ]
                )

            accuracies = []  # Store accuracy for each classifier in this run

            classifiers = [
                (SVC(kernel="linear"), "SVC_linear"),
                (KNN(n_neighbors=5), "KNeighborsClassifier"),
                (LR(max_iter=1000), "LogisticRegression"),
                (RF(n_estimators=100), "RandomForestClassifier"),
                (
                    MLP(
                        hidden_layer_sizes=(
                            50,
                            50,
                            25,
                        ),
                        max_iter=1000,
                        random_state=42,
                    ),
                    "MLPClassifier_3Layer",
                ),
            ]

            # --------------------------------------------------------------------------------------------------------------------------------------
            summary_text = "\n"
            # %% Classifiers loop
            for clf, clf_name in classifiers:
                # Train the classifier with encoded labels
                clf.fit(train_features, train_labels_encoded)

                if clf_name == "RandomForestClassifier":
                    tree_depths = [
                        estimator.tree_.max_depth for estimator in clf.estimators_
                    ]
                    max_depth_of_forest = max(tree_depths)

                cv_scores = cross_val_score(
                    clf, train_features, train_labels_encoded, cv=5, scoring="accuracy"
                )
                summary_text += f"""
        {clf_name:<25s}| CV-Accuracy: {100 * np.mean(cv_scores):6.2f}% ± {100 * np.std(cv_scores):.2f}%"""

                # Test the classifier
                predictions = clf.predict(test_features)

                # Convert predictions back to original labels for reporting
                predictions_labels = label_encoder.inverse_transform(predictions)

                # Generate confusion matrix
                cm = confusion_matrix(test_labels, predictions_labels)
                print(cm)

                # Generate classification report as dictionary
                cr = classification_report(
                    test_labels, predictions_labels, output_dict=True, zero_division=0
                )
                accuracy = cr["accuracy"]
                precision = cr["weighted avg"]["precision"]
                recall = cr["weighted avg"]["recall"]
                f1 = cr["weighted avg"]["f1-score"]
                fpr = (
                    1 - cr["pristine"]["recall"]
                )  # False positive rate = 1 - recall for pristine class

                # Store all metrics in the nested structure
                performance_metrics[clf_name]["accuracy"].append(accuracy)
                performance_metrics[clf_name]["precision"].append(precision)
                performance_metrics[clf_name]["recall"].append(recall)
                performance_metrics[clf_name]["f1"].append(f1)
                performance_metrics[clf_name]["fpr"].append(fpr)

                accuracies.append((clf_name, accuracy))

                # Model details
                if clf_name.startswith("SVC_linear"):
                    n_features = train_features.shape[1]
                    X_var = train_features.var()
                    gamma_scale = 1 / (n_features * X_var)
                    model_details = f"SVC_linear: kernel={clf.kernel}, C={clf.C}, gamma = {gamma_scale}, support_vectors={clf.support_vectors_.shape[0] if hasattr(clf, 'support_vectors_') else 'N/A'}"

                elif clf_name == "KNeighborsClassifier":
                    model_details = f"KNN: n_neighbors={clf.n_neighbors}, weights={clf.weights}, params={clf.get_params()}, features={train_features.shape[1]}"
                elif clf_name == "LogisticRegression":
                    model_details = f"LR: C={clf.C}, params={clf.get_params()}, features={train_features.shape[1]}"
                elif clf_name == "RandomForestClassifier":
                    avg_depth = np.mean(
                        [
                            tree.get_depth()
                            for tree in clf.estimators_
                            if hasattr(tree, "get_depth")
                        ]
                    )
                    model_details = f"RF: params={clf.get_params()}, n_estimators={clf.n_estimators}, max_deph = {max_depth_of_forest}, avg_depth={avg_depth:.2f}, features={train_features.shape[1]}"
                elif clf_name == "MLPClassifier_3Layer":
                    model_details = f"MLP: hidden_layers={clf.hidden_layer_sizes}, iterations={clf.n_iter_}, features={train_features.shape[1]}"
                else:
                    model_details = "No Model Details Available"

                print(f"Classifier: {clf_name}")
                print(f"\tModel Details: {model_details}")

                print(
                    f"\tCV-ACC  : {100 * np.mean(cv_scores):6.2f}% ± {100 * np.std(cv_scores):.2f}%"
                )
                print(f"\tTEST-ACC: {100 * accuracy:6.2f}%\n")

            min_accuracy = min(accuracies, key=lambda x: x[1])
            max_accuracy = max(accuracies, key=lambda x: x[1])
            min_classifier, min_acc_value = min_accuracy
            max_classifier, max_acc_value = max_accuracy

            print(
                f"\nRun {run} Minimum Accuracy: {100 * min_acc_value:6.2f}% (Classifier: {min_classifier})"
            )
            print(
                f"Run {run} Maximum Accuracy: {100 * max_acc_value:6.2f}% (Classifier: {max_classifier})\n"
            )

        summary_text += f"""\n
        Results Summary:"""
        for clf_name in performance_metrics:
            summary_text += f"""\n
        -------------------- {clf_name:^25s} --------------------\n"""
            for metric_name, values in performance_metrics[clf_name].items():
                min_val = min(values) if values else 0
                avg_val = np.mean(values) if values else 0
                std_val = np.std(values) if values else 0
                max_val = max(values) if values else 0

                multiplier = 100

                summary_text += f"""
        {metric_name.capitalize():<12s}| Min: {multiplier * min_val:6.2f} | Avg: {multiplier * avg_val:6.2f} | Std: {multiplier * std_val:6.2f} | Max: {multiplier * max_val:6.2f}"""
        summary_text += "\n"

        print(summary_text)
        print("\n")

        with open(log_file, "a") as f:
            f.write(summary_text)
        print(
            f"Analysis Summary for Sample Length {sample_length} Saved to -\n{log_file}\n"
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = str(timedelta(seconds=int(elapsed_time)))
    print(f"Total Elapsed time: {formatted_time} (hh:mm:ss)\n\n")
