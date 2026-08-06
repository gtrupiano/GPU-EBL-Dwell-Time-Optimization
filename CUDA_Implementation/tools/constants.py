###############################################################################
# File Name: constants.py
# Description:
# Shared paths and algorithm values used by CUDA utility scripts.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

from pathlib import Path


###############################################################################
# CONSTANTS
###############################################################################

TOOLS_FOLDER_PATH = Path(__file__).resolve().parent
CUDA_IMPLEMENTATION_FOLDER_PATH = TOOLS_FOLDER_PATH.parent

CUDA_INPUT_DATA_FOLDER_PATH = CUDA_IMPLEMENTATION_FOLDER_PATH / "input_data"
CUDA_OUTPUT_DATA_FOLDER_PATH = CUDA_IMPLEMENTATION_FOLDER_PATH / "output_data"

MASK_SIZE = 65
MAX_DWELL_TIME = 2.0