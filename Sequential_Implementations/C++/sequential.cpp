/*
**********************************************************************
    File Name: sequential.cpp
    Description: Sequential CPU implementation of the dwell-time
    optimization algorithm used by the CUDA implementation.
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
#include <stdlib.h>
#include <string.h>

/*
 **********************************************************************
 * DEFINES and CONSTANTS
 **********************************************************************
*/

typedef unsigned int uint;

#define TARGET_LAYOUT_ARG_INDEX 0
#define PSF_MASK_ARG_INDEX 1

// Algorithm Parameters
const uint MAX_ITERATIONS = 100;
const float MINIMUM_MSE = 0.001f;
const float LEARNING_RATE = 10.0f;
const float LEARNING_RATE_DECAY = 0.99f;
const float LEARNING_RATE_MINIMUM = 0.1f;
const float MAX_DWELL_TIME = 5.0f;

// Frequency of MSE Iteration Logging
const uint MSE_ITERATION_LOG_INTERVAL = 5;

// Must match the CUDA convolution mask width.
const uint CONVOLUTION_MASK_WIDTH = 65;
const uint CONVOLUTION_MASK_RADIUS = CONVOLUTION_MASK_WIDTH / 2;

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

// Sequential Intermediate Data
float *depositedEnergy;
float *errorMatrix;
float *dwellTimeCorrection;
float *bestDwellTimeMap;

// Sequential Output Data
float *dwellTimeMap;
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
static void initializeData(void);
static void runOptimization(void);
static void copyResultsToHost(void);
static void exportOutput(void);
static void freeMemory(void);

// Sequential Algorithm Functions
static void convolveImage(
    const float *inputImage,
    const float *psfMask,
    uint imageWidth,
    uint imageHeight,
    float *outputImage
);

static float calculateError(
    const float *targetLayout,
    const float *depositedEnergyInput,
    uint imageWidth,
    uint imageHeight,
    float *outputErrorMatrix
);

static void updateDwellTime(
    const float *inputErrorMatrix,
    const float *psfMask,
    uint imageWidth,
    uint imageHeight,
    float learningRate,
    float maxDwellTime,
    float *outputDwellTimeCorrection,
    float *inputOutputDwellTimeMap
);

/*
 **********************************************************************
 * GLOBAL FUNCTIONS
 **********************************************************************
*/

/**************************************************
 * Function: main
 * Description: Handles program initialization,
 * sequential component execution, result retrieval,
 * output, and cleanup.
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

    wbTime_start(Compute, "CPU Memory Allocation");
    // Allocate memory for all sequential working arrays
    allocateMemory();
    wbTime_stop(Compute, "CPU Memory Allocation");

    wbTime_start(Copy, "Initializing CPU Data");
    // Initialize dwell time map using the target layout
    initializeData();
    wbTime_stop(Copy, "Initializing CPU Data");

    wbTime_start(Compute, "Optimization Algorithm");
    // Optimization Loop
    runOptimization();
    wbTime_stop(Compute, "Optimization Algorithm");

    wbTime_start(Copy, "Copying Results");
    // Copy the best result into the output buffer
    copyResultsToHost();
    wbTime_stop(Copy, "Copying Results");

    wbTime_start(Copy, "Exporting Data To File ");
    // Saving output data to file
    exportOutput();
    wbTime_stop(Copy, "Exporting Data To File ");

    wbTime_start(Compute, "CPU Memory Deallocation");
    // Deallocate memory
    freeMemory();
    wbTime_stop(Compute, "CPU Memory Deallocation");

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

    // Obtaining dimensional data from image
    targetLayoutWidth = wbImage_getWidth(targetLayoutImage);
    targetLayoutHeight = wbImage_getHeight(targetLayoutImage);
    targetLayoutChannels = wbImage_getChannels(targetLayoutImage);

    // Calculating size of image
    targetLayoutSizeBytes = targetLayoutWidth * targetLayoutHeight * sizeof(float);

    // Reading image and storing data in proper variable
    hostTargetLayout = wbImage_getData(targetLayoutImage);

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
    bool psfRowsInvalid = psfMaskRows != (int)CONVOLUTION_MASK_WIDTH;
    bool psfColsInvalid = psfMaskColumns != (int)CONVOLUTION_MASK_WIDTH;

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
 * Description: Allocates CPU memory required for
 * all sequential working variables.
**************************************************/

static void allocateMemory(void)
{
    depositedEnergy = (float *)malloc(targetLayoutSizeBytes);
    errorMatrix = (float *)malloc(targetLayoutSizeBytes);
    dwellTimeCorrection = (float *)malloc(targetLayoutSizeBytes);
    bestDwellTimeMap = (float *)malloc(targetLayoutSizeBytes);
    dwellTimeMap = (float *)malloc(targetLayoutSizeBytes);
}

/**************************************************
 * Function: initializeData
 * Description: Initializes the dwell time map from
 * the target layout, matching the CUDA implementation.
**************************************************/

static void initializeData(void)
{
    memcpy(dwellTimeMap, hostTargetLayout, targetLayoutSizeBytes);
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
    float currentLearningRate = LEARNING_RATE;

    for(uint iteration = 0; iteration < MAX_ITERATIONS; iteration++)
    {
        // Deposited energy calculation using current dwell time map
        convolveImage(
            dwellTimeMap,
            hostPsfMask,
            targetLayoutWidth,
            targetLayoutHeight,
            depositedEnergy
        );

        // Calculate error matrix and mean squared error (MSE)
        mse = calculateError(
            hostTargetLayout,
            depositedEnergy,
            targetLayoutWidth,
            targetLayoutHeight,
            errorMatrix
        );

        // Determine whether current MSE is lowest
        // If so then store off MSE, current dwell time map and current iteration number
        if(mse < bestMSE)
        {
            bestMSE = mse;
            bestIteration = iteration;

            memcpy(bestDwellTimeMap, dwellTimeMap, targetLayoutSizeBytes);
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
            errorMatrix,
            hostPsfMask,
            targetLayoutWidth,
            targetLayoutHeight,
            currentLearningRate,
            MAX_DWELL_TIME,
            dwellTimeCorrection,
            dwellTimeMap
        );

        // Reduce the learning rate by 1% after each dwell time update.
        // Larger steps are used early in optimization, while later iterations
        // become progressively more conservative as the solution converges.
        currentLearningRate *= LEARNING_RATE_DECAY;

        // Constrain learning rate to minimum
        if(currentLearningRate < LEARNING_RATE_MINIMUM)
        {
            currentLearningRate = LEARNING_RATE_MINIMUM;
        }
    }

    // After all iterations, log best MSE calculated
    wbLog(TRACE, "Best MSE: ", bestMSE, " at Iteration: ", (bestIteration + 1));
}

/**************************************************
 * Function: copyResultsToHost
 * Description: Allocates the output buffer and
 * copies the best sequential dwell time map into it.
**************************************************/

static void copyResultsToHost(void)
{
    hostDwellTimeMap = (float *)malloc(targetLayoutSizeBytes);

    memcpy(hostDwellTimeMap, bestDwellTimeMap, targetLayoutSizeBytes);
}

/**************************************************
 * Function: exportOutput
 * Description: Exports the optimized dwell time map
 * to the specified output file.
**************************************************/

static void exportOutput(void)
{
    wbExport(
        outputDwellTimeFile,
        hostDwellTimeMap,
        targetLayoutHeight,
        targetLayoutWidth
    );
}

/**************************************************
 * Function: freeMemory
 * Description: Deallocates CPU memory as well as
 * deletes the imported target-layout image.
**************************************************/

static void freeMemory(void)
{
    free(depositedEnergy);
    free(errorMatrix);
    free(dwellTimeCorrection);
    free(bestDwellTimeMap);
    free(dwellTimeMap);
    free(hostDwellTimeMap);

    // Delete libwb image object
    wbImage_delete(targetLayoutImage);

    // Free imported PSF mask data
    free(hostPsfMask);
}

/**************************************************
 * Function: convolveImage
 * Description: Sequentially convolves the input
 * image with the PSF using the same zero-padding
 * behavior as the CUDA convolution implementation.
**************************************************/

static void convolveImage(
    const float *inputImage,
    const float *psfMask,
    uint imageWidth,
    uint imageHeight,
    float *outputImage
)
{
    for(uint outputRow = 0; outputRow < imageHeight; outputRow++)
    {
        for(uint outputCol = 0; outputCol < imageWidth; outputCol++)
        {
            float convolutionOutput = 0.0f;

            for(uint maskRow = 0; maskRow < CONVOLUTION_MASK_WIDTH; maskRow++)
            {
                for(uint maskCol = 0; maskCol < CONVOLUTION_MASK_WIDTH; maskCol++)
                {
                    int inputRow = (int)outputRow + (int)maskRow - (int)CONVOLUTION_MASK_RADIUS;
                    int inputCol = (int)outputCol + (int)maskCol - (int)CONVOLUTION_MASK_RADIUS;

                    bool inputRowValid = inputRow >= 0 && inputRow < (int)imageHeight;
                    bool inputColValid = inputCol >= 0 && inputCol < (int)imageWidth;

                    if(inputRowValid && inputColValid)
                    {
                        uint inputIdx = ((uint)inputRow * imageWidth) + (uint)inputCol;
                        uint maskIdx = (maskRow * CONVOLUTION_MASK_WIDTH) + maskCol;

                        convolutionOutput += inputImage[inputIdx] * psfMask[maskIdx];
                    }
                }
            }

            uint outputIdx = (outputRow * imageWidth) + outputCol;
            outputImage[outputIdx] = convolutionOutput;
        }
    }
}

/**************************************************
 * Function: calculateError
 * Description: Sequentially calculates the same
 * target-minus-deposited error matrix and MSE used
 * by the CUDA implementation.
**************************************************/

static float calculateError(
    const float *targetLayout,
    const float *depositedEnergyInput,
    uint imageWidth,
    uint imageHeight,
    float *outputErrorMatrix
)
{
    float squaredErrorSum = 0.0f;
    uint numberOfPixels = imageWidth * imageHeight;

    for(uint pixelIdx = 0; pixelIdx < numberOfPixels; pixelIdx++)
    {
        float error = targetLayout[pixelIdx] - depositedEnergyInput[pixelIdx];

        outputErrorMatrix[pixelIdx] = error;
        squaredErrorSum += error * error;
    }

    return squaredErrorSum / (float)numberOfPixels;
}

/**************************************************
 * Function: updateDwellTime
 * Description: Sequentially calculates the dwell
 * time correction and applies the same cubic,
 * learning-rate-scaled update and clamp as CUDA.
**************************************************/

static void updateDwellTime(
    const float *inputErrorMatrix,
    const float *psfMask,
    uint imageWidth,
    uint imageHeight,
    float learningRate,
    float maxDwellTime,
    float *outputDwellTimeCorrection,
    float *inputOutputDwellTimeMap
)
{
    // Calculate the dwell time correction:
    // dwellTimeCorrection = error matrix convolved with PSF
    convolveImage(
        inputErrorMatrix,
        psfMask,
        imageWidth,
        imageHeight,
        outputDwellTimeCorrection
    );

    uint numberOfPixels = imageWidth * imageHeight;

    for(uint pixelIdx = 0; pixelIdx < numberOfPixels; pixelIdx++)
    {
        float currentDwell = inputOutputDwellTimeMap[pixelIdx];
        float currentDwellTimeCorrection = outputDwellTimeCorrection[pixelIdx];

        // Apply cubic sensitivity to the dwell-time correction.
        // Cubing preserves the correction direction while suppressing
        // small corrections and emphasizing larger correction magnitudes.
        float sensitiveDwellTimeCorrection =
            currentDwellTimeCorrection *
            currentDwellTimeCorrection *
            currentDwellTimeCorrection;

        // Scale the correction by the current learning rate and
        // apply it to the existing dwell time.
        float updatedDwell =
            currentDwell + (sensitiveDwellTimeCorrection * learningRate);

        // Clamp the updated dwell time to the allowed exposure range.
        if(updatedDwell < 0.0f)
        {
            updatedDwell = 0.0f;
        }
        else if(updatedDwell > maxDwellTime)
        {
            updatedDwell = maxDwellTime;
        }

        inputOutputDwellTimeMap[pixelIdx] = updatedDwell;
    }
}
