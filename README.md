# PRIDES-2026
Artifact Repository for PRIDES submission to USENIX-2026

Included is the datasets directory, containing zip files with the raw power trace outputs as well as converted PRIDES of each of the benchmarks and their test cases. The uncompressed size is roughly 10 GB.

The scripts directory contains PRIDES_ML.py and TestAllClassificationData.py. TestAllClassificationData will take in parameters for the type of benchmark being performed, as well as other various conditions such as downsampling. The input and output directories need to be modified to match the benchmark as well as what condition you want to simulate, i.e. using ideal 3.3V data for train and 3.0V or temperature-varying data for test. An output file will be produced in the output directory displaying the results from the various classifiers. Evaluating either the power trace or PRIDES data will both utilize the raw power trace data, and will be converted at runtime to the correct format.
