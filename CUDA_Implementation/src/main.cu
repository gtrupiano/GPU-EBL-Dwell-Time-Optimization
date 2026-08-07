/*
**********************************************************************
	File Name: main.cu
	Description:
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
uint sizeOfTargetLayout;

// These need to be int due to wbImport data type inputs
int psfMaskRows;
int psfMaskColumns;
uint sizeOfPsfMask;


// Input
float *hostTargetLayout;
float *hostPsfMask;

float *deviceTargetLayout;
float *devicePsfMask;

// Intermediate Pointers
float *deviceDepositedEnergy;
float *deviceErrorMatrix;
float *deviceSquaredErrorSum; // A single value, not a matrix
float *deviceDwellTimeCorrection;
float *deviceBestDwellTimeMap;

// Output
float *hostDwellTimeMap;

float *deviceDwellTimeMap;

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

static void cudaCheck(cudaError_t error);
static void loadInputs(wbArg_t args);
static bool verifyInputs(void);
static void allocateMemory(void);
static void copyDataToDevice(void);
static void runOptimization(void);
static void copyResultsToHost(void);
static void freeMemory(void);

/*
 **********************************************************************
 * GLOBAL FUNCTIONS
 **********************************************************************
*/


/**************************************************
 * Function: main
 * Description: 
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
 * Description: 
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
    sizeOfTargetLayout = targetLayoutWidth * targetLayoutHeight * sizeof(float);

    // Reading image and porting data to proper variables
    hostTargetLayout  = wbImage_getData(targetLayoutImage);


    // PSF Mask data procurement

    // Load PSF mask as a matrix
    hostPsfMask = (float *)wbImport(psfMaskFile, &psfMaskRows, &psfMaskColumns);

    // Calculating size of matrix
    sizeOfPsfMask = psfMaskRows * psfMaskColumns * sizeof(float);
}


/**************************************************
 * Function: verifyInputs
 * Description: 
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
 * Description: 
**************************************************/

static void allocateMemory(void)
{
    // Input Pointers
    cudaCheck(cudaMalloc((void **)&deviceTargetLayout, sizeOfTargetLayout));
    cudaCheck(cudaMalloc((void **)&devicePsfMask, sizeOfPsfMask));
    
    // Intermediate Pointers
    cudaCheck(cudaMalloc((void **)&deviceDepositedEnergy, sizeOfTargetLayout));
    cudaCheck(cudaMalloc((void **)&deviceErrorMatrix, sizeOfTargetLayout));
    cudaCheck(cudaMalloc((void **)&deviceSquaredErrorSum, sizeof(float)));
    cudaCheck(cudaMalloc((void **)&deviceDwellTimeCorrection, sizeOfTargetLayout));
    cudaCheck(cudaMalloc((void **)&deviceBestDwellTimeMap, sizeOfTargetLayout));

    // Output Pointers
    cudaCheck(cudaMalloc((void **)&deviceDwellTimeMap, sizeOfTargetLayout));
}


/**************************************************
 * Function: copyDataToDevice
 * Description: 
**************************************************/
static void copyDataToDevice(void)
{
    cudaCheck(cudaMemcpy(deviceTargetLayout, hostTargetLayout, sizeOfTargetLayout, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(devicePsfMask, hostPsfMask, sizeOfPsfMask, cudaMemcpyHostToDevice));

    // Initialize dwell-time map using the target layout
    cudaCheck(cudaMemcpy(deviceDwellTimeMap, deviceTargetLayout, sizeOfTargetLayout, cudaMemcpyDeviceToDevice));
}


/**************************************************
 * Function: runOptimization
 * Description: 
**************************************************/

static void runOptimization(void)
{
    float mse = 0.0f;
    float bestMSE = FLT_MAX;
    uint bestIteration = 0;

    for(uint iteration = 0; iteration < MAX_ITERATIONS; iteration++)
    {
        // Deposited energy calculation
        convolveImage(
            deviceDwellTimeMap,
            targetLayoutWidth,
            targetLayoutHeight,
            devicePsfMask,
            deviceDepositedEnergy
        );

        // Calculate error matrix and mean squared error (MSE)
        mse = calculateError(
            deviceTargetLayout,
            deviceDepositedEnergy,
            targetLayoutWidth,
            targetLayoutHeight,
            deviceErrorMatrix,
            deviceSquaredErrorSum
        );

        // Determine whether current MSE is lowest
        // If so then store off MSE, current dwell time map and current iteration number
        if(mse < bestMSE)
        {
            bestMSE = mse;
            bestIteration = iteration;

            cudaCheck(cudaMemcpy(deviceBestDwellTimeMap, deviceDwellTimeMap, sizeOfTargetLayout, cudaMemcpyDeviceToDevice));
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

        // Calculate the dwell time
        updateDwellTime(
            deviceDwellTimeMap,
            deviceErrorMatrix,
            devicePsfMask,
            deviceDwellTimeCorrection,
            targetLayoutWidth,
            targetLayoutHeight,
            LEARNING_RATE,
            MAX_DWELL_TIME
        );
    }

    // After all iterations, log of best MSE calculated
    wbLog(TRACE, "Best MSE: ", bestMSE, " at Iteration: ", (bestIteration + 1));
}


/**************************************************
 * Function: copyResultsToHost
 * Description: 
**************************************************/

static void copyResultsToHost(void)
{
    hostDwellTimeMap = (float *)malloc(sizeOfTargetLayout);

    // Copy from device to host
    cudaCheck(cudaMemcpy(hostDwellTimeMap, deviceBestDwellTimeMap, sizeOfTargetLayout, cudaMemcpyDeviceToHost));

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
 * Description: 
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
 * Description: Helper function for verifying CUDA
 * API's executed properly
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