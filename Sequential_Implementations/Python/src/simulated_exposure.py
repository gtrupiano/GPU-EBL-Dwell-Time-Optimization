###############################################################################
# File Name: simulated_exposure.py
# Description:
# Simulates deposited electron-beam energy and iteratively adjusts a
# continuous relative dwell-time map to reduce exposure error.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

# Library Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
import time

# File Imports
import constants


###############################################################################
# GLOBAL VARIABLES
###############################################################################


###############################################################################
# GLOBAL FUNCTIONS
###############################################################################

def main():
    # Captures and processes arguments
    args = parse_arguments()

    # Load the IC layout and convert it to binary values: black = 0, white = 1.
    ic_image = Image.open(args.ic_path).convert("1")
    ic_layout = np.asarray(ic_image, dtype=np.float64)

    # Load the same libwb PSF mask used by the CUDA and C++ implementations.
    psf_mask = read_libwb_raw(args.psf_path)

    # Initially, expose the same locations as the desired IC layout.

    # Unlike ic_layout, dwell_time_map will become grayscale
    dwell_time_map = ic_layout.copy()

    # Capturing start time for optimization duration
    start_optimization_time = time.perf_counter()

    # Iteratively adjust the dwell-time map and record the MSE after each iteration.
    mse_history, best_mse, best_iteration, best_dwell_time_map = optimize_dwell_time(
        ic_layout,
        psf_mask,
        dwell_time_map
    )

    # Capturing end time for optimization duration
    end_optimization_time = time.perf_counter()

    # Calculating optimization duration (and converting from sec to ms)
    optimization_time = end_optimization_time - start_optimization_time
    optimization_time *= 1000

    # Use the best dwell-time map as the final output.
    dwell_time_map[:, :] = best_dwell_time_map

    # Used for benchmarking
    print(f"Optimization Time (ms): {optimization_time:.6f}")
    print(f"Best MSE: {best_mse:.12e} at Iteration: {best_iteration}")

    # Saving dwell time output
    np.save(
        args.dwell_output_path,
        dwell_time_map
    )

     # Only calculate/display visualization data when requested
    if args.show_plots:
        # Recalculate the final deposited energy after the final dwell-time update.
        deposited_energy = fftconvolve(
            dwell_time_map,
            psf_mask,
            mode="same"
        )

        error_matrix, mse = calculate_error_matrix(
            ic_layout,
            deposited_energy
        )

        show_plots(
            ic_layout,
            dwell_time_map,
            psf_mask,
            deposited_energy,
            error_matrix,
            mse,
            mse_history
        )


###############################################################################
# Function Name: parse_arguments
# Description:
###############################################################################

def parse_arguments():
    # Parser object declaration
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("ic_path")
    parser.add_argument("psf_path")
    parser.add_argument("dwell_output_path")

    # Optional argument for showing plots
    parser.add_argument(
        "--show-plots",
        action="store_true"
    )

    return parser.parse_args()


###############################################################################
# Function Name: read_libwb_raw
# Description:
###############################################################################

def read_libwb_raw(input_path):
    with open(input_path, "r", encoding="utf-8") as file:
        header = file.readline().split()

        rows = int(header[0])
        columns = int(header[1])

        values = np.array(
            file.read().split(),
            dtype=np.float32
        )

    expected_values = rows * columns

    if values.size != expected_values:
        raise ValueError(
            f"Expected {expected_values} PSF values, "
            f"but found {values.size}."
        )

    return values.reshape(rows, columns)


###############################################################################
# Function Name: calculate_error_matrix
# Description:
###############################################################################

def calculate_error_matrix(ic_layout, deposited_energy):
    error_matrix = np.zeros_like(deposited_energy)
    error_sum = 0.0

    # Fetching real image size
    image_height, image_width = ic_layout.shape

    for row in range(image_height):
        for col in range(image_width):
            # Desired exposure minus actual deposited energy.
            error_matrix[row][col] = (ic_layout[row][col] - deposited_energy[row][col])

            # Calculate squared error.
            error_value_squared = error_matrix[row][col] ** 2

            # Add the squared error to the total.
            error_sum += error_value_squared

    # Calculate the mean squared error.
    mse = error_sum / (image_height * image_width)

    return error_matrix, mse


###############################################################################
# Function Name: optimize_dwell_time
# Description:
###############################################################################

def optimize_dwell_time(ic_layout, psf_mask, dwell_time_map):
    mse_history = []
    best_mse = float("inf") # Python trick to do infinitly large number
    best_iteration = 0
    best_dwell_time_map = None
    current_learning_rate = constants.LEARNING_RATE

    image_height, image_width = dwell_time_map.shape

    for iteration in range(constants.MAX_ITERATIONS):
        # Simulate the deposited energy produced by the current dwell-time map (this is what's changing)
        deposited_energy = fftconvolve(
            dwell_time_map,
            psf_mask,
            mode="same"
        )

        # Calculate the error between the desired exposure and actual
        # deposited energy.
        error_matrix, mse = calculate_error_matrix(
            ic_layout,
            deposited_energy
        )

        mse_history.append(mse)

        # Keeping track of the best mse map
        if mse < best_mse:
            best_mse = mse
            best_iteration = iteration + 1
            best_dwell_time_map = dwell_time_map.copy()

        # Only logging MSE at specified interval
        if (iteration == 0) or ((iteration + 1) % constants.MSE_ITERATION_LOG_INTERVAL == 0):
            print(
                f"Iteration: {iteration + 1}; "
                f"MSE = {mse:.12e}"
            )

        # Stop early when the MSE becomes sufficiently small.
        if(mse <= constants.MINIMUM_MSE):
            break
        
        # Convolve the error with the PSF to determine which dwell-time pixels
        # contributed to each exposure error. This accounts for energy spreading
        # into neighboring pixels instead of updating only the same pixel.
        dwell_time_correction = fftconvolve(
            error_matrix,
            psf_mask,
            mode="same"
        )

        # Update the grayscale dwell-time map.

        # Positive error:
        #   Desired energy is greater than deposited energy.
        #   Increase dwell time.

        # Negative error:
        #   Deposited energy is greater than desired energy.
        #   Decrease dwell time.

        # Update each dwell-time pixel using its corresponding error value.
        for row in range(image_height):
            for col in range(image_width):
                current_dwell_time_correction = dwell_time_correction[row][col]

                sensitive_dwell_time_correction = (current_dwell_time_correction * current_dwell_time_correction * current_dwell_time_correction)
                dwell_time_map[row][col] += (sensitive_dwell_time_correction * current_learning_rate)

                # Keep the dwell time within its allowed range.
                if dwell_time_map[row][col] < 0.0:
                    dwell_time_map[row][col] = 0.0

                elif dwell_time_map[row][col] > constants.MAX_DWELL_TIME:
                    dwell_time_map[row][col] = constants.MAX_DWELL_TIME

        current_learning_rate *= constants.LEARNING_RATE_DECAY

        if current_learning_rate < constants.LEARNING_RATE_MINIMUM:
            current_learning_rate = constants.LEARNING_RATE_MINIMUM

    return mse_history, best_mse, best_iteration, best_dwell_time_map


###############################################################################
# Function Name: show_plots
# Description:
###############################################################################

def show_plots(
    ic_layout,
    dwell_time_map,
    psf_mask,
    deposited_energy,
    error_matrix,
    mse,
    mse_history
):
    plt.figure()
    plt.imshow(ic_layout, cmap="gray", vmin=0, vmax=1)
    plt.title("Desired IC Layout")
    plt.colorbar()

    plt.figure()
    plt.imshow(
        dwell_time_map,
        cmap="gray",
        vmin=0,
        vmax=constants.MAX_DWELL_TIME
    )
    plt.title("Optimized Relative Dwell-Time Map")
    plt.colorbar(label="Relative dwell time")

    plt.figure()

    # Prevent log10(0).
    log_mask = np.log10(
        np.maximum(psf_mask, np.finfo(np.float64).tiny)
    )

    plt.imshow(log_mask, cmap="hot")
    plt.title("PSF Mask")
    plt.colorbar()

    plt.figure()
    plt.imshow(deposited_energy, cmap="hot", vmin=0, vmax=1)
    plt.title("Deposited Energy")
    plt.colorbar()

    plt.figure()
    plt.imshow(error_matrix ** 2, cmap="hot", vmin=0, vmax=1)
    plt.title(f"Squared Error — MSE: {mse:.6f}")
    plt.colorbar(label="Squared exposure error")

    plt.figure()
    plt.plot(mse_history)
    plt.title("MSE During Dwell-Time Optimization")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()