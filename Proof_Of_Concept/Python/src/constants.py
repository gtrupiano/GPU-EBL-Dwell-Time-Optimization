from pathlib import Path

###############################################################################
# CONSTANTS
###############################################################################

# Folder locations
SRC_FOLDER_PATH = Path(__file__).resolve().parent
PYTHON_FOLDER_PATH = SRC_FOLDER_PATH.parent

INPUT_DATA_FOLDER_PATH = PYTHON_FOLDER_PATH / "input_data"
DATA_OUTPUT_FOLDER_PATH = PYTHON_FOLDER_PATH / "output_data"


# Double-Gaussian parameters (Later ALPHA, BETA, and N will be set to the values from the selections below):
# Note: Values are from paper

# 25 kV, 1um HSQ
ALPHA_25KV_1UM = 0.2263
BETA_25KV_1UM = 2.9986
N_25KV_1UM = 1.1191

# 100 kV, 1um HSQ
ALPHA_100KV_1UM = 0.0482
BETA_100KV_1UM = 29.7343
N_100KV_1UM = 0.7237

# 100 kV, 50nm HSQ
ALPHA_100KV_50NM = 0.0024
BETA_100KV_50NM = 27.8728
N_100KV_50NM = 0.7183

# Range of forward scattering
ALPHA = ALPHA_100KV_1UM

# Range of backward scattering
BETA = BETA_100KV_1UM

# Ratio of total energy deposited
N = N_100KV_1UM


# The 128x128 layout represents a 1 µm x 1 µm area.
IMAGE_WIDTH_UM = 1.0
IMAGE_SIZE_PIXELS = 128

# Physical distance represented by one image pixel.
PIXEL_SIZE_UM = IMAGE_WIDTH_UM / IMAGE_SIZE_PIXELS

# The CUDA paper uses a 65x65 PSF mask.
MASK_SIZE = 65

# Input paths
IC_IMAGE_PATH = INPUT_DATA_FOLDER_PATH / "IC128.png"

# PSF output paths
PSF_1D_OUTPUT_IMAGE_PATH = DATA_OUTPUT_FOLDER_PATH / "double_gaussian_psf_1D.png"
PSF_2D_OUTPUT_DATA_PATH = DATA_OUTPUT_FOLDER_PATH / "double_gaussian_psf_2D.npy"
PSF_2D_OUTPUT_IMAGE_PATH = DATA_OUTPUT_FOLDER_PATH / "double_gaussian_psf_2D.png"

# Exposure simulation output paths
DEPOSITED_ENERGY_OUTPUT_DATA_PATH = DATA_OUTPUT_FOLDER_PATH / "deposited_energy.npy"
DEPOSITED_ENERGY_OUTPUT_IMAGE_PATH = DATA_OUTPUT_FOLDER_PATH / "deposited_energy.png"