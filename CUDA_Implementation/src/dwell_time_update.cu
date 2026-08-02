/*
**********************************************************************
	File Name: dwell_time_update.cu
	Description:
**********************************************************************
*/

/*
 **********************************************************************
 * INCLUDES
 **********************************************************************
*/

#include "dwell_time_update.cuh"
#include "convolution.cuh"

/*
 **********************************************************************
 * DEFINES and CONSTANTS
 **********************************************************************
*/


/*
 **********************************************************************
 * GLOBAL VARIABLES
 **********************************************************************
*/

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

static __global__ void dwellTimeUpdateKernel(
    float *deviceDwellTimeMap,
    const float *deviceDwellTimeCorrection,
    uint imageWidth,
    uint imageHeight,
    float learningRate,
    float maxDwellTime
);

/*
 **********************************************************************
 * GLOBAL FUNCTIONS
 **********************************************************************
*/

/**************************************************
 * Kernel: updateDwellTime
 * Description: 
**************************************************/

void updateDwellTime(
    float *deviceDwellTimeMap,
    const float *deviceErrorMap,
    const float *devicePsfMask,
    float *deviceDwellTimeCorrection,
    uint imageWidth,
    uint imageHeight,
    float learningRate,
    float maxDwellTime
)
{
    // Calculate the dwell-time correction:
    // dwellTimeCorrection = errorMap * PSF

    convolveImage(
        deviceErrorMap,
        imageWidth,
        imageHeight,
        devicePsfMask,
        deviceDwellTimeCorrection
    );

    // Apply the correction to the dwell-time map.

    // Threads per block value for each dimension used
    dim3 dwellBlockSize(DWELL_BLOCK_WIDTH, DWELL_BLOCK_HEIGHT, 1);

    // Calculate grid size (number of blocks)
	uint dwellBlocksPerDimX = (imageWidth + DWELL_BLOCK_WIDTH - 1) / DWELL_BLOCK_WIDTH;
	uint dwellBlocksPerDimY = (imageHeight + DWELL_BLOCK_HEIGHT - 1) / DWELL_BLOCK_HEIGHT;

	// establish number of blocks
	dim3 dwellNumberOfBlocks(dwellBlocksPerDimX, dwellBlocksPerDimY, 1);

	// launch kernel in host wrapper code
	dwellTimeUpdateKernel<<<dwellNumberOfBlocks, dwellBlockSize>>>(
        deviceDwellTimeMap,
        deviceDwellTimeCorrection,
        imageWidth,
        imageHeight,
        learningRate,
        maxDwellTime
    );
}


/**************************************************
 * Kernel: dwellTimeUpdateKernel
 * Description: 
**************************************************/

static __global__ void dwellTimeUpdateKernel(	
	float *deviceDwellTimeMap, 
	const float *deviceDwellTimeCorrection, 
	uint imageWidth, 
	uint imageHeight, 
	float learningRate, 
	float maxDwellTime
)
{
	// find the 2D coordinates
	uint rowIdx = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint colIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

	// convert to 1D
	uint globalPixelIdx1D = colIdx + rowIdx * imageWidth;

	// boundary checking
	if(rowIdx < imageHeight && colIdx < imageWidth)
    {  
        float currentDwell = deviceDwellTimeMap[globalPixelIdx1D];
        float dwellTimeCorrection = deviceDwellTimeCorrection[globalPixelIdx1D];

        float updatedDwell = currentDwell + (dwellTimeCorrection * learningRate);
        
        if(updatedDwell < 0.0f)
        {
            updatedDwell = 0.0f;

        }
        else if(updatedDwell > maxDwellTime)
        {
            updatedDwell = maxDwellTime;
        }

        deviceDwellTimeMap[globalPixelIdx1D] = updatedDwell;
    } 
}