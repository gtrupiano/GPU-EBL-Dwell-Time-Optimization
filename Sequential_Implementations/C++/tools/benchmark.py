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
from pathlib import Path
import re
import subprocess
import time

###############################################################################
# CONSTANTS
###############################################################################

# Base Directories
TOOLS_DIRECTORY = Path(__file__).resolve().parent
CPP_DIRECTORY = TOOLS_DIRECTORY.parent
REPOSITORY_DIRECTORY = CPP_DIRECTORY.parent.parent

# Use the same input data as the CUDA implementation
INPUT_DATA_DIRECTORY = REPOSITORY_DIRECTORY / "CUDA_Implementation" / "input_data"

BENCHMARK_RESULTS_DIRECTORY = TOOLS_DIRECTORY / "benchmark_results"

# Executable Path
SEQUENTIAL_EXECUTABLE = CPP_DIRECTORY / "build" / "Sequential_EBL_Dwell_Time_Optimization"

# Script Path
RAW_TO_PNG_SCRIPT = REPOSITORY_DIRECTORY / "CUDA_Implementation" / "tools" / "convert_raw_to_png.py"

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

# Benchmarking Parameters
# How many times the same dataset is executed. This is used for the average amount of execution time
NUMBER_OF_TRIALS = 10

# Algorithm parameters from sequential.cpp (UPDATE FROM sequential.cpp)
MAX_ITERATIONS = 100
MINIMUM_MSE = 0.001
LEARNING_RATE = 10.0
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_MINIMUM = 0.1
MAX_DWELL_TIME = 5.0

###############################################################################
# GLOBAL VARIABLES
###############################################################################


###############################################################################
# GLOBAL FUNCTIONS
###############################################################################

def main():
    # Create benchmark output directory if it doesn't exist already
    BENCHMARK_RESULTS_DIRECTORY.mkdir(exist_ok=True)

    dataset_outputs = capture_dataset_output()

    benchmark_results, mse_history = process_dataset_results(dataset_outputs)

    write_results(benchmark_results, mse_history)


###############################################################################
# Function Name: capture_dataset_output
# Description:
###############################################################################

def capture_dataset_output():
    dataset_outputs = []

    for dataset_name, ic_path, psf_path in DATASETS:
        print("Running dataset:", dataset_name)

        # Unique output path name based on inputs
        dwell_output_path = BENCHMARK_RESULTS_DIRECTORY / f"{dataset_name}_dwell.raw"

        # This will get overwritten each trial but that's okay since the result will be the same.
        # Only used to hold a trial's results
        trial_results = None

        # Execute the same dataset multiple times to get average timing measurements
        total_program_time_sum = 0
        total_optimization_time_sum = 0

        for trial in range(NUMBER_OF_TRIALS):
            print(f"Trial {trial + 1}/{NUMBER_OF_TRIALS}")

            # Capture start time to check how long the whole sequential program takes to execute
            start_time = time.perf_counter()

            # Pass data into input arguments of sequential executable
            output = subprocess.check_output(
                [
                    str(SEQUENTIAL_EXECUTABLE),
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

            # Capture end time
            end_time = time.perf_counter()

            # Calculate total time to check how long the whole program takes to execute
            execution_time = end_time - start_time

            # Add this trial's execution time to the sum for later averaging calculations
            total_program_time_sum += execution_time

            # Fetching all needed data from JSON output
            trial_results = get_parsed_output(output)
            optimization_time = trial_results["optimization_time"]

            # Verify optimization time parameter was present in JSON output
            if optimization_time is None:
                raise RuntimeError(f"Optimization time not found for {dataset_name}, trial {trial + 1}")

            # Adding this trial's optimization time to the sum for later averaging calculations
            total_optimization_time_sum += optimization_time

        # Calculate the average execution time for this dataset (converts from sec to ms)
        average_execution_time = total_program_time_sum / NUMBER_OF_TRIALS
        average_execution_time *= 1000

        # Calculate the average optimization time for this dataset
        average_optimization_time = total_optimization_time_sum / NUMBER_OF_TRIALS

        # Convert dwell-time output from .raw to .png for additional data collection
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
            "dwell_output_png_path": dwell_output_png_path,
            "average_execution_time": average_execution_time,
            "average_optimization_time": average_optimization_time,
            "parsed_output": trial_results
        })

    return dataset_outputs


###############################################################################
# Function Name: get_parsed_output
# Description:
###############################################################################

def get_parsed_output(output):
    optimization_time = None
    best_mse = None
    best_iteration = None
    mse_history = []

    # Parse output from sequential program
    for line in output.splitlines():
        # Load data into JSON object
        try:
            log_data = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Look for JSON headers
        data = log_data.get("data", {})
        message = data.get("message", "")

        # Get optimization execution time
        if message == "Optimization Algorithm":
            elapsed_time = data.get("elapsed_time")

            if elapsed_time is not None:
                # Convert from ns to ms
                optimization_time = elapsed_time / 1000000.0

        # Get logged MSE values
        mse_match = re.match(r"Iteration: (\d+); MSE = ([0-9.eE+-]+)", message)

        if mse_match:
            mse_history.append({
                "Iteration": int(mse_match.group(1)),
                "MSE": float(mse_match.group(2))
            })

        # Get best MSE and iteration
        best_mse_match = re.match(r"Best MSE: ([0-9.eE+-]+) at Iteration: (\d+)", message)

        if best_mse_match:
            best_mse = float(best_mse_match.group(1))
            best_iteration = int(best_mse_match.group(2))

    parsed_output = {
        "optimization_time": optimization_time,
        "best_mse": best_mse,
        "best_iteration": best_iteration,
        "mse_history": mse_history
    }

    return parsed_output


###############################################################################
# Function Name: process_dataset_results
# Description:
###############################################################################

def process_dataset_results(dataset_results):
    benchmark_results = []
    mse_history = []

    for result in dataset_results:
        # Fetch results already parsed during execution
        parsed_results = result["parsed_output"]

        # Helpers for datastructure of dataset_results and its sub datastructure parsed_results
        dataset_name = result["dataset_name"]
        ic = result["ic_path"].name
        psf = result["psf_path"].name
        average_optimization_time = result["average_optimization_time"]
        average_execution_time = result["average_execution_time"]
        dwell_output_path = result["dwell_output_path"].name

        best_mse = parsed_results["best_mse"]
        best_iteration = parsed_results["best_iteration"]
        original_mse_history = parsed_results["mse_history"]

        # Verify needed results were present in JSON output
        if (best_mse is None) or (best_iteration is None):
            raise RuntimeError(f"Needed data was not found for {dataset_name}")

        # Add this dataset's MSE history
        for mse_result in original_mse_history:
            mse_history.append({
                "Dataset": dataset_name,
                "Iteration": mse_result["Iteration"],
                "MSE": mse_result["MSE"]
            })

        # Append benchmark results
        benchmark_results.append({
            "Dataset": dataset_name,
            "IC": ic,
            "PSF": psf,

            "Max_Iterations": MAX_ITERATIONS,
            "Minimum_MSE": MINIMUM_MSE,
            "Learning_Rate": LEARNING_RATE,
            "Learning_Rate_Decay": LEARNING_RATE_DECAY,
            "Learning_Rate_Minimum": LEARNING_RATE_MINIMUM,
            "Max_Dwell_Time": MAX_DWELL_TIME,

            "Average_Optimization_Time_(ms)": average_optimization_time,
            "Average_Program_Time_(ms)": average_execution_time,
            "Number_of_Trials": NUMBER_OF_TRIALS,

            "Best_MSE": best_mse,
            "Best_Iteration": best_iteration,
            "Dwell_Time_Output": dwell_output_path
        })

    return benchmark_results, mse_history


###############################################################################
# Function Name: write_results
# Description:
###############################################################################

def write_results(benchmark_results, mse_history):
    # Write benchmark results
    with open(BENCHMARK_RESULTS_PATH, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Dataset",
                "IC",
                "PSF",

                "Max_Iterations",
                "Minimum_MSE",
                "Learning_Rate",
                "Learning_Rate_Decay",
                "Learning_Rate_Minimum",
                "Max_Dwell_Time",

                "Average_Optimization_Time_(ms)",
                "Average_Program_Time_(ms)",
                "Number_of_Trials",

                "Best_MSE",
                "Best_Iteration",
                "Dwell_Time_Output"
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