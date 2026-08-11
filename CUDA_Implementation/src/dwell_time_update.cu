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
 * LOCAL FUNCTION / KERNEL PROTOTYPES (declare as static)
 **********************************************************************
*/

static __global__ void dwellTimeUpdateKernel(
    // Inputs
    const float *dwellTimeCorrection,
    uint imageWidth,
    uint imageHeight,
    float learningRate,
    float maxDwellTime,

    // Input / Output
    float *dwellTimeMap
);

/*
 **********************************************************************
 * GLOBAL FUNCTIONS
 **********************************************************************
*/

/**************************************************
 * Function: updateDwellTime
 * Description: 
**************************************************/

void updateDwellTime(
    // Inputs
    const float *deviceErrorMatrix,
    const float *devicePsfMask,
    uint imageWidth,
    uint imageHeight,
    float learningRate,
    float maxDwellTime,
    
    // Intermediate
    float *deviceDwellTimeCorrection,
    
    // Input / Output
    float *deviceDwellTimeMap
)
{
    // Calculate the dwell time correction:
    // dwellTimeCorrection = error matrix convolved with PSF
    convolveImage(
        deviceErrorMatrix,
        devicePsfMask,
        imageWidth,
        imageHeight,
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

	// Launch kernel in host wrapper code
    // Dwell time map is updated in kernel
	dwellTimeUpdateKernel<<<dwellNumberOfBlocks, dwellBlockSize>>>(
        deviceDwellTimeCorrection,
        imageWidth,
        imageHeight,
        learningRate,
        maxDwellTime,
        deviceDwellTimeMap
    );
}


/**************************************************
 * Kernel: dwellTimeUpdateKernel
 * Description: 
**************************************************/
static __global__ void dwellTimeUpdateKernel(
    // Inputs
    const float *dwellTimeCorrection,
    uint imageWidth,
    uint imageHeight,
    float learningRate,
    float maxDwellTime,

    // Input / Output
    float *dwellTimeMap
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
        float currentDwell = dwellTimeMap[globalPixelIdx1D];
        float currentDwellTimeCorrection = dwellTimeCorrection[globalPixelIdx1D];

        float updatedDwell = currentDwell + (currentDwellTimeCorrection * learningRate);
        
        if(updatedDwell < 0.0f)
        {
            updatedDwell = 0.0f;

        }
        else if(updatedDwell > maxDwellTime)
        {
            updatedDwell = maxDwellTime;
        }

        dwellTimeMap[globalPixelIdx1D] = updatedDwell;
    } 
}