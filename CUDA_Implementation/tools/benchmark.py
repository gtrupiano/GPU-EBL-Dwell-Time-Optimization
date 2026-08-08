###############################################################################
# File Name: benchmark.py
# Description:
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

# Library Imports
import csv
import json
import re
import subprocess
from pathlib import Path


###############################################################################
# CONSTANTS
###############################################################################

# Base Directories
TOOLS_DIRECTORY = Path(__file__).resolve().parent
CUDA_DIRECTORY = TOOLS_DIRECTORY.parent
INPUT_DATA_DIRECTORY = CUDA_DIRECTORY / "input_data"
BENCHMARK_RESULTS_DIRECTORY = TOOLS_DIRECTORY / "benchmark_results"

# Executable Path
CUDA_EXECUTABLE = CUDA_DIRECTORY / "build" / "GPU_EBL_Dwell_Time_Optimization"

# Script Path
RAW_TO_PNG_SCRIPT = TOOLS_DIRECTORY / "convert_raw_to_png.py"

# IC Layout Paths
IC128_PATH = INPUT_DATA_DIRECTORY / "IC128.ppm"
IC256_PATH = INPUT_DATA_DIRECTORY / "IC256.ppm"
IC512_PATH = INPUT_DATA_DIRECTORY / "IC512.ppm"

# PSF Mask Paths
PSF_25KV_1UM_HSQ_PATH = INPUT_DATA_DIRECTORY / "PSF_Mask_25kV_1um-HSQ.raw"
PSF_100KV_1UM_HSQ_PATH = INPUT_DATA_DIRECTORY / "PSF_Mask_100kV_1um-HSQ.raw"
PSF_100KV_50NM_HSQ_PATH = INPUT_DATA_DIRECTORY / "PSF_Mask_100kV_50nm-HSQ.raw"


# Result Paths
BENCHMARK_RESULTS_PATH = BENCHMARK_RESULTS_DIRECTORY / "benchmark_results.csv"

MSE_HISTORY_PATH = BENCHMARK_RESULTS_DIRECTORY / "mse_history.csv"


# Benchmark Datasets
DATASETS = [
    ("IC128_25kV_1um-HSQ", IC128_PATH, PSF_25KV_1UM_HSQ_PATH),
    ("IC128_100kV_1um-HSQ", IC128_PATH, PSF_100KV_1UM_HSQ_PATH),
    ("IC128_100kV_50nm-HSQ", IC128_PATH, PSF_100KV_50NM_HSQ_PATH),

    ("IC256_25kV_1um-HSQ", IC256_PATH, PSF_25KV_1UM_HSQ_PATH),
    ("IC256_100kV_1um-HSQ", IC256_PATH, PSF_100KV_1UM_HSQ_PATH),
    ("IC256_100kV_50nm-HSQ", IC256_PATH, PSF_100KV_50NM_HSQ_PATH),

    ("IC512_25kV_1um-HSQ", IC512_PATH, PSF_25KV_1UM_HSQ_PATH),
    ("IC512_100kV_1um-HSQ", IC512_PATH, PSF_100KV_1UM_HSQ_PATH),
    ("IC512_100kV_50nm-HSQ", IC512_PATH, PSF_100KV_50NM_HSQ_PATH),
]


###############################################################################
# GLOBAL VARIABLES
###############################################################################


###############################################################################
# GLOBAL FUNCTIONS
###############################################################################

def main():
    # Create benchmark output directory if it doesn't exist already
    BENCHMARK_RESULTS_DIRECTORY.mkdir(exist_ok=True)

    dataset_outputs = capture_dataset_results()

    benchmark_results, mse_history = parse_results(dataset_outputs)

    write_results(benchmark_results, mse_history)


###############################################################################
# Function Name: capture_dataset_results
# Description:
###############################################################################

def capture_dataset_results():
    dataset_outputs = []

    for dataset_name, ic_path, psf_path in DATASETS:
        print("Running dataset:", dataset_name)
        
        # Unique output path name based on inputs
        dwell_output_path = BENCHMARK_RESULTS_DIRECTORY / f"{dataset_name}_dwell.raw"

        # Pass data into input arguments of CUDA executable
        output = subprocess.check_output(
            [
                str(CUDA_EXECUTABLE),
                "-i",
                f"{ic_path},{psf_path}",
                "-o",
                str(dwell_output_path),
                "-t",
                "image"
            ],
            stderr=subprocess.STDOUT,
            text=True
        )

        # Convert dwell-time output from .raw to .png for additional data collection (useful for output analysis)
        dwell_output_png_path = BENCHMARK_RESULTS_DIRECTORY / f"{dataset_name}_dwell.png"

        subprocess.check_output(
            [
                "python3",
                str(RAW_TO_PNG_SCRIPT),
                str(dwell_output_path),
                "-o",
                str(dwell_output_png_path)
            ],
            stderr=subprocess.STDOUT,
            text=True
        )

        # Save results in array for later analysis
        dataset_outputs.append({
            "dataset_name": dataset_name,
            "ic_path": ic_path,
            "psf_path": psf_path,
            "dwell_output_path": dwell_output_path,
            "output": output
        })

    return dataset_outputs


###############################################################################
# Function Name: parse_results
# Description:
###############################################################################

def parse_results(dataset_results):
    benchmark_results = []
    mse_history = []

    for result in dataset_results:
        # Helpers for datastructure of dataset_results
        dataset_name = result["dataset_name"]
        dataset_output = result["output"]

        # Initialize variables incase messages don't show up in output
        optimization_time = None
        best_mse = None
        best_iteration = None

        # Parse output from CUDA program
        for line in dataset_output.splitlines():
            try:
                log_data = json.loads(line)
            except json.JSONDecodeError:
                continue

            data = log_data.get("data", {})
            message = data.get("message", "")

            # Get optimization execution time
            if message == "Optimization Algorithm":
                elapsed_time = data.get("elapsed_time")

                if elapsed_time is not None:
                    # Convert from ns to ms for readability
                    optimization_time = elapsed_time / 1000000.0

            # Get logged MSE values
            mse_match = re.match(r"Iteration: (\d+); MSE = ([0-9.eE+-]+)", message)

            if mse_match:
                mse_history.append({
                    "Dataset": dataset_name,
                    "Iteration": int(mse_match.group(1)),
                    "MSE": float(mse_match.group(2))
                })

            # Get best MSE and iteration
            best_mse_match = re.match(
                r"Best MSE: ([0-9.eE+-]+) at Iteration: (\d+)",
                message
            )

            if best_mse_match:
                best_mse = float(best_mse_match.group(1))
                best_iteration = int(best_mse_match.group(2))

        benchmark_results.append({
            "Dataset": dataset_name,
            "IC": result["ic_path"].name,
            "PSF": result["psf_path"].name,
            "Optimization Time (ms)": optimization_time,
            "Best MSE": best_mse,
            "Best Iteration": best_iteration,
            "Dwell Time Output": result["dwell_output_path"].name
        })

    return benchmark_results, mse_history


###############################################################################
# Function Name: write_results
# Description:
###############################################################################

def write_results(benchmark_results, mse_history):
        # Write benchmark results
    with open(BENCHMARK_RESULTS_PATH, "w", newline="") as file:
        # Declare each header before writing data
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Dataset",
                "IC",
                "PSF",
                "Optimization Time (ms)",
                "Best MSE",
                "Best Iteration",
                "Dwell Time Output"
            ]
        )

        writer.writeheader()
        writer.writerows(benchmark_results)

    # Write MSE history
    with open(MSE_HISTORY_PATH, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Dataset",
                "Iteration",
                "MSE"
            ]
        )

        writer.writeheader()
        writer.writerows(mse_history)


if __name__ == "__main__":
    main()