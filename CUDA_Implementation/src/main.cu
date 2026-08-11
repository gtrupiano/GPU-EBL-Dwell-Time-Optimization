/*
**********************************************************************
	File Name: main.cu
	Description: Manages program input, GPU memory, iterative dwell
    time optimization algorithm, result transfer, and output.
**********************************************************************
*/

/*
 **********************************************************************
 * INCLUDES
 **********************************************************************
*/

// Libraries
#include <wb.h>
#include <float.h>

// Sources
#include "convolution.cuh"
#include "error_calculations.cuh"
#include "dwell_time_update.cuh"

/*
 **********************************************************************
 * DEFINES and CONSTANTS
 **********************************************************************
*/

#define TARGET_LAYOUT_ARG_INDEX 0
#define PSF_MASK_ARG_INDEX 1

// Algorithm Parameters
const uint MAX_ITERATIONS = 1000;
const float MINIMUM_MSE = 0.001f;
const float LEARNING_RATE = 0.1f;
const float MAX_DWELL_TIME = 2.0f;

// Frequency of MSE Iteration Logging
const uint MSE_ITERATION_LOG_INTERVAL = 50;

/*
 **********************************************************************
 * GLOBAL VARIABLES
 **********************************************************************
*/

// File Variables
char *targetLayoutFile;
char *psfMaskFile;
char *outputDwellTimeFile;

// Image Variables
wbImage_t targetLayoutImage;

uint targetLayoutWidth;
uint targetLayoutHeight;
uint targetLayoutChannels;
uint targetLayoutSizeBytes;

// These need to be int due to wbImport data type inputs
int psfMaskRows;
int psfMaskColumns;
uint psfMaskSizeBytes;


// Host Input Data
float *hostTargetLayout;
float *hostPsfMask;

// Device Input Data
float *deviceTargetLayout;
float *devicePsfMask;

// Device Intermediate Data
float *deviceDepositedEnergy;
float *deviceErrorMatrix;
float *deviceSquaredErrorSum; // A single value, not a matrix
float *deviceDwellTimeCorrection;
float *deviceBestDwellTimeMap;

// Device Output Data
float *deviceDwellTimeMap;

// Host Output Data
float *hostDwellTimeMap;

/*
 **********************************************************************
 * LOCAL TYPES
 **********************************************************************
*/

/*
 **********************************************************************
 * LOCAL VARIABLES (declare as static)
 **********************************************************************
*/

/*
 **********************************************************************
 * LOCAL FUNCTION PROTOTYPES (declare as static)
 **********************************************************************
*/

// Main Program Functions
static void loadInputs(wbArg_t args);
static bool verifyInputs(void);
static void allocateMemory(void);
static void copyDataToDevice(void);
static void runOptimization(void);
static void copyResultsToHost(void);
static void exportOutput(void);
static void freeMemory(void);

// Utility Function
static void cudaCheck(cudaError_t error);

/*
 **********************************************************************
 * GLOBAL FUNCTIONS
 **********************************************************************
*/


/**************************************************
 * Function: main
 * Description: Handles program initialization, CUDA
 * component execution, result retrieval, output,
 * and cleanup.
**************************************************/

int main(int argc, char **argv) 
{
    wbArg_t args = wbArg_read(argc, argv);

    // Loading all input arguments
    loadInputs(args);

    // Verify whether all input data is the proper size
    bool inputsValid = verifyInputs();
    
    if(!inputsValid)
    {
        // Libwb error logs are done within verifyInputs function
        return -1;
    }

    wbTime_start(GPU, "GPU Memory Allocation");
    // Allocate memory for all device pointers
    allocateMemory();
    wbTime_stop(GPU, "GPU Memory Allocation");

    
    wbTime_start(Copy,"Copying Data To GPU");
    // Copy inputs to GPU
    copyDataToDevice();
    wbTime_stop(Copy, "Copying Data To GPU");


    wbTime_start(Compute, "Optimization Algorithm");
    // Optimization Loop
    runOptimization();
    wbTime_stop(Compute, "Optimization Algorithm");


    wbTime_start(Copy,"Copying Data From GPU");
    // Transfer data from device to host
    copyResultsToHost();
    wbTime_stop(Copy,"Copying Data From GPU");


    wbTime_start(Copy, "Exporting Data To File ");
    // Saving output data to file
    exportOutput();
    wbTime_stop(Copy, "Exporting Data To File ");
    

    wbTime_start(GPU, "GPU Memory Deallocation");
    // Deallocate memory
    freeMemory();
    wbTime_stop(GPU, "GPU Memory Deallocation");


    return 0;
}

/*
 **********************************************************************
 * LOCAL FUNCTIONS
 **********************************************************************
*/


/**************************************************
 * Function: loadInputs
 * Description: Loads the target layout and PSF mask
 * from the input argument files and stores their
 * dimensions.
**************************************************/

static void loadInputs(wbArg_t args)
{
    // Obtaining files from input arguments
    targetLayoutFile = wbArg_getInputFile(args, TARGET_LAYOUT_ARG_INDEX);
    psfMaskFile = wbArg_getInputFile(args, PSF_MASK_ARG_INDEX);

    // Obtaining file from output argument
    outputDwellTimeFile = wbArg_getOutputFile(args);

    // Target layout data procurement

    // Load target layout image into proper variables.
    targetLayoutImage = wbImport(targetLayoutFile);

    // Obtaining dimentional data from images
    targetLayoutWidth = wbImage_getWidth(targetLayoutImage);
    targetLayoutHeight = wbImage_getHeight(targetLayoutImage);
    targetLayoutChannels = wbImage_getChannels(targetLayoutImage);

    // Calculating size of image
    targetLayoutSizeBytes = targetLayoutWidth * targetLayoutHeight * sizeof(float);

    // Reading image and porting data to proper variables
    hostTargetLayout  = wbImage_getData(targetLayoutImage);


    // PSF Mask data procurement

    // Load PSF mask as a matrix
    hostPsfMask = (float *)wbImport(psfMaskFile, &psfMaskRows, &psfMaskColumns);

    // Calculating size of matrix
    psfMaskSizeBytes = psfMaskRows * psfMaskColumns * sizeof(float);
}


/**************************************************
 * Function: verifyInputs
 * Description: Verifies that the loaded target 
 * layout and PSF mask meet the required input
 * dimensions and format.
**************************************************/

static bool verifyInputs(void)
{
    // Make sure target layout is grayscale
    if(targetLayoutChannels != 1)
    {
        wbLog(ERROR, "Target layout must contain one channel.");

        return false;
    }

    // Check whether rows and columns for PSF mask match expected
    bool psfRowsInvalid = psfMaskRows != CONVOLUTION_MASK_WIDTH;
    bool psfColsInvalid = psfMaskColumns != CONVOLUTION_MASK_WIDTH;

    if(psfRowsInvalid || psfColsInvalid)
    {
        wbLog(ERROR, "PSF mask must be ", CONVOLUTION_MASK_WIDTH, "x", CONVOLUTION_MASK_WIDTH);
    
        return false;
    }

    // All conditions passed so return true
    return true;
}


/**************************************************
 * Function: allocateMemory
 * Description: Allocates GPU memory required for
 * all needed device variables.
**************************************************/

static void allocateMemory(void)
{
    // Input Pointers
    cudaCheck(cudaMalloc((void **)&deviceTargetLayout, targetLayoutSizeBytes));
    cudaCheck(cudaMalloc((void **)&devicePsfMask, psfMaskSizeBytes));
    
    // Intermediate Pointers
    cudaCheck(cudaMalloc((void **)&deviceDepositedEnergy, targetLayoutSizeBytes));
    cudaCheck(cudaMalloc((void **)&deviceErrorMatrix, targetLayoutSizeBytes));
    cudaCheck(cudaMalloc((void **)&deviceSquaredErrorSum, sizeof(float)));
    cudaCheck(cudaMalloc((void **)&deviceDwellTimeCorrection, targetLayoutSizeBytes));
    cudaCheck(cudaMalloc((void **)&deviceBestDwellTimeMap, targetLayoutSizeBytes));

    // Output Pointers
    cudaCheck(cudaMalloc((void **)&deviceDwellTimeMap, targetLayoutSizeBytes));
}


/**************************************************
 * Function: copyDataToDevice
 * Description: Copies the input data to GPU memory
 * and initializes the dwell time map from the
 * target layout.
**************************************************/
static void copyDataToDevice(void)
{
    cudaCheck(cudaMemcpy(deviceTargetLayout, hostTargetLayout, targetLayoutSizeBytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(devicePsfMask, hostPsfMask, psfMaskSizeBytes, cudaMemcpyHostToDevice));

    // Initialize dwell time map using the target layout
    cudaCheck(cudaMemcpy(deviceDwellTimeMap, deviceTargetLayout, targetLayoutSizeBytes, cudaMemcpyDeviceToDevice));
}


/**************************************************
 * Function: runOptimization
 * Description: Iteratively calculates deposited
 * energy and error, updates dwell times, and tracks
 * the lowest-MSE result.
**************************************************/

static void runOptimization(void)
{
    float mse = 0.0f;
    float bestMSE = FLT_MAX;
    uint bestIteration = 0;

    for(uint iteration = 0; iteration < MAX_ITERATIONS; iteration++)
    {
        // Deposited energy calculation using current dwell time map
        convolveImage(
            deviceDwellTimeMap,
            devicePsfMask,
            targetLayoutWidth,
            targetLayoutHeight,
            deviceDepositedEnergy
        );

        // Calculate error matrix and mean squared error (MSE)
        mse = calculateError(
            deviceTargetLayout,
            deviceDepositedEnergy,
            targetLayoutWidth,
            targetLayoutHeight,
            deviceSquaredErrorSum,
            deviceErrorMatrix
        );

        // Determine whether current MSE is lowest
        // If so then store off MSE, current dwell time map and current iteration number
        if(mse < bestMSE)
        {
            bestMSE = mse;
            bestIteration = iteration;

            cudaCheck(cudaMemcpy(deviceBestDwellTimeMap, deviceDwellTimeMap, targetLayoutSizeBytes, cudaMemcpyDeviceToDevice));
        }

        // Logging data for iteration number and MSE value at these intervals:
        // - First iteration (to capture where it started)
        // - Every MSE_ITERATION_LOG_INTERVAL iteration
        bool firstIteration = iteration == 0;
        bool mseIterationInterval = (iteration + 1) % MSE_ITERATION_LOG_INTERVAL == 0;

        if(firstIteration || mseIterationInterval)
        {
            wbLog(TRACE, "Iteration: ", (iteration + 1), "; MSE = ", mse);
        }

        // Preemptively stop optimization if the MSE is too low
        if(mse <= MINIMUM_MSE)
        {
            break;
        }

        // Update the dwell time map using the current error
        updateDwellTime(
            deviceErrorMatrix,
            devicePsfMask,
            targetLayoutWidth,
            targetLayoutHeight,
            LEARNING_RATE,
            MAX_DWELL_TIME,
            deviceDwellTimeCorrection,
            deviceDwellTimeMap
        );
    }

    // After all iterations, log of best MSE calculated
    wbLog(TRACE, "Best MSE: ", bestMSE, " at Iteration: ", (bestIteration + 1));
}


/**************************************************
 * Function: copyResultsToHost
 * Description: Allocates host memory and copies the
 * best dwell time map from GPU memory to host memory.
**************************************************/

static void copyResultsToHost(void)
{
    hostDwellTimeMap = (float *)malloc(targetLayoutSizeBytes);

    // Copy from device to host
    cudaCheck(cudaMemcpy(hostDwellTimeMap, deviceBestDwellTimeMap, targetLayoutSizeBytes, cudaMemcpyDeviceToHost));
}


/**************************************************
 * Function: exportOutput
 * Description: Exports the optimized dwell time map
 * from host memory to the specified output file.
**************************************************/

static void exportOutput(void)
{
    // Store the results from host to a file
    wbExport(
        outputDwellTimeFile,
        hostDwellTimeMap,
        targetLayoutHeight,
        targetLayoutWidth
    );
}


/**************************************************
 * Function: freeMemory
 * Description: Deallocates GPU and host memory as
 * well as deletes the imported target-layout image.
**************************************************/

static void freeMemory(void)
{
    // Input Pointers
    cudaCheck(cudaFree(deviceTargetLayout));
    cudaCheck(cudaFree(devicePsfMask));

    // Intermediate Pointers
    cudaCheck(cudaFree(deviceDepositedEnergy));
    cudaCheck(cudaFree(deviceErrorMatrix));
    cudaCheck(cudaFree(deviceSquaredErrorSum));
    cudaCheck(cudaFree(deviceDwellTimeCorrection));
    cudaCheck(cudaFree(deviceBestDwellTimeMap));

    // Output Pointers
    cudaCheck(cudaFree(deviceDwellTimeMap));

    // Delete libwb image objects
    wbImage_delete(targetLayoutImage);

    // Free host side variables
    free(hostPsfMask);
    free(hostDwellTimeMap);
}


/**************************************************
 * Function: cudaCheck
 * Description: Checks output of CUDA runtime API
 * and exits the program if an error occurred.
**************************************************/

static void cudaCheck(cudaError_t error)
{
    if(error != cudaSuccess)
    {
        wbLog(ERROR, "Failed to run command ");
        wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(error));

        // Exits the program upon call
        exit(EXIT_FAILURE);
    }
}