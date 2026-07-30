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
#include <wb.h>
#include "dwell_time_update.cuh"

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
__global__ void dwellTimeUpdateKernel(	
	float *deviceDwellTimeMap, 
	const float *deviceErrorMap, 
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
void updateDwellTime(
	float *deviceDwellTimeMap, 
	const float *deviceErrorMap, 
	uint imageWidth, 
	uint imageHeight, 
	float learningRate, 
	float maxDwellTime
)
{
    // Threads per block value for each dimension used
    dim3 blockSize(DWELL_BLOCK_WIDTH, DWELL_BLOCK_HEIGHT, 1);

    // Calculate grid size (number of blocks)
	uint blocksPerDimX = (imageWidth + DWELL_BLOCK_WIDTH - 1) / DWELL_BLOCK_WIDTH;
	uint blocksPerDimY = (imageHeight + DWELL_BLOCK_HEIGHT - 1) / DWELL_BLOCK_HEIGHT;

	// establish number of blocks
	dim3 numberOfBlocks(blocksPerDimX, blocksPerDimY, 1);

	// launch kernel in host wrapper code
	dwellTimeUpdateKernel<<<numberOfBlocks, blockSize>>>(
		deviceDwellTimeMap, 
		deviceErrorMap, 
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

__global__ void dwellTimeUpdateKernel(	
	float *deviceDwellTimeMap, 
	const float *deviceErrorMap, 
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
	if(rowIdx < imageHeight && colIdx < imageWidth){  

            float currentDwell = deviceDwellTimeMap[globalPixelIdx1D];

            float error = deviceErrorMap[globalPixelIdx1D];

            float updatedDwell = currentDwell + (error * learningRate);

            if(updatedDwell < 0.0f){

                updatedDwell = 0.0f;

            }else if(updatedDwell > maxDwellTime){

                updatedDwell = maxDwellTime;

            }

			deviceDwellTimeMap[globalPixelIdx1D] = updatedDwell;

        } 







}