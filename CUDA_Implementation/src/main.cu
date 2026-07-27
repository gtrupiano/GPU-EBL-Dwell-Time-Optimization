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
 
#include <wb.h>
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

static void calculateDepositedEnergy(
    float *icLayout,
    int imageWidth,
    int imageHeight,
    const float *devicePsfMask,
    float *deviceDepositedEnergy
);

/*
 **********************************************************************
 * GLOBAL FUNCTIONS
 **********************************************************************
*/


/**************************************************
 * Function: 
 * Description: 
**************************************************/

int main(int argc, char **argv) 
{
    wbArg_t args;

    args = wbArg_read(argc, argv);

    wbLog(TRACE, "Test");
}


/**************************************************
 * Function: calculateDepositedEnergy
 * Description: 
**************************************************/

static void calculateDepositedEnergy(
    float *icLayout,
    int imageWidth,
    int imageHeight,
    const float *devicePsfMask,
    float *deviceDepositedEnergy
)
{
    // Threads per block value for each dimention used
    dim3 blockSize(CONVOLUTION_OUTPUT_TILE_WIDTH, CONVOLUTION_OUTPUT_TILE_WIDTH, 1);

	// Grid size
    // There needs to be enough threads for each row,col of the matrix.
    // So by dividing length of matrix / threadsPerBlock, it should yield number of blocks.
    // Rounding up is needed (integer based rounding
    uint blocksPerDimX = (imageWidth + CONVOLUTION_OUTPUT_TILE_WIDTH - 1) / CONVOLUTION_OUTPUT_TILE_WIDTH;
    uint blocksPerDimY = (imageHeight + CONVOLUTION_OUTPUT_TILE_WIDTH - 1) / CONVOLUTION_OUTPUT_TILE_WIDTH;
    dim3 numberOfBlocks(blocksPerDimX, blocksPerDimY, 1);

    convolutionKernel<<<numberOfBlocks, blockSize>>>(
      icLayout,
      imageWidth,
      imageHeight,
      devicePsfMask,
      deviceDepositedEnergy
   );
}