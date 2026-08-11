/*
**********************************************************************
   File Name: convolution.cu
   Description:
**********************************************************************
*/

/*
**********************************************************************
* INCLUDES
**********************************************************************
*/

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

static __global__ void convolutionKernel(
    // Inputs
    const float *inputImage,
    const float *mask,
    uint imageWidth,
    uint imageHeight,

    // Output
    float *outputImage
);

/*
**********************************************************************
* GLOBAL FUNCTIONS
**********************************************************************
*/

/**************************************************
* Kernel: convolveImage
* Description:
**************************************************/

void convolveImage(
    // Inputs
    const float *deviceInputImage,
    const float *devicePsfMask,
    uint imageWidth,
    uint imageHeight,

    // Output
    float *deviceOutputImage
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
        deviceInputImage,
        devicePsfMask,
        imageWidth,
        imageHeight,
        deviceOutputImage
    );
}


/*
 **********************************************************************
 * LOCAL FUNCTIONS
 **********************************************************************
*/


/**************************************************
* Kernel: convolutionKernel
* Description:
**************************************************/

static __global__ void convolutionKernel(
    // Inputs
    const float *inputImage,
    const float *mask,
    uint imageWidth,
    uint imageHeight,

    // Output
    float *outputImage
)
{
    // In order to compute convolution, additional values are needed in a given tile
    // due to the mask needing CONVOLUTION_MASK_RADIUS values on both the first and last elements
    // of the inputImage.
    __shared__ float inputTile[CONVOLUTION_INPUT_TILE_WIDTH][CONVOLUTION_INPUT_TILE_WIDTH];

    int tileRow = threadIdx.y;
    int tileCol = threadIdx.x;

    // Convert the 2D tile index into a 1D
    int tileIdx1D = (tileRow * blockDim.x) + tileCol;

    // Total number of threads in block.
    int numThreadsPerBlock = blockDim.x * blockDim.y;

    // Total number of elements that need to be loaded into shared memory.
    int neededInputElements = CONVOLUTION_INPUT_TILE_WIDTH * CONVOLUTION_INPUT_TILE_WIDTH;


    // Since there are more input tiles than threads, having each thread load
    // multiple input elements is needed.

    // Starting at whatever the thread is running, it will load N elements from input incrementing by
    // number of threads per block as to not overlap local threads that are already doing this action.
    // This will continue until the needed amount of elements are loaded (cooporative loading)
    for(int inputTileIdx = tileIdx1D; inputTileIdx < neededInputElements; inputTileIdx += numThreadsPerBlock)
    {
        // Converting the inputTileIdx back to 2D for the shared memory
        // Row is whole tile width increments while col is partial increments
        int inputTileRow = inputTileIdx / CONVOLUTION_INPUT_TILE_WIDTH;
        int inputTileCol = inputTileIdx % CONVOLUTION_INPUT_TILE_WIDTH;

        // CONVOLUTION_OUTPUT_TILE_WIDTH is used for the block stride since each block
        // advances by 16x16 output elements, not by the extra elements needed for convolution.
        int inputImageRow = (int)(blockIdx.y * CONVOLUTION_OUTPUT_TILE_WIDTH) + inputTileRow - (int)CONVOLUTION_MASK_RADIUS;
        int inputImageCol = (int)(blockIdx.x * CONVOLUTION_OUTPUT_TILE_WIDTH) + inputTileCol - (int)CONVOLUTION_MASK_RADIUS;


        // Conditional parameters for whether input tile can be populated
        bool imageInputRowBoundaryValid = inputImageRow >= 0 && inputImageRow < imageHeight;
        bool imageInputColBoundaryValid = inputImageCol >= 0 && inputImageCol < imageWidth;
        bool imageInputBoundaryValid = imageInputRowBoundaryValid && imageInputColBoundaryValid;

        if(imageInputBoundaryValid)
        {
            uint inputImageIdx = (inputImageRow * imageWidth) + inputImageCol;

            inputTile[inputTileRow][inputTileCol] = inputImage[inputImageIdx];
        }
        else
        {
            inputTile[inputTileRow][inputTileCol] = 0.0f;
        }
    }

    // Waiting for shared memory to be populated with all input data needed for convolution
    __syncthreads();


    // CONVOLUTION_OUTPUT_TILE_WIDTH is used for the block stride since each block
    // advances by 16x16 output elements, not by the extra elements needed for convolution.
    int outputImageRow = (blockIdx.y * CONVOLUTION_OUTPUT_TILE_WIDTH) + tileRow;
    int outputImageCol = (blockIdx.x * CONVOLUTION_OUTPUT_TILE_WIDTH) + tileCol;

    // Conditional parameters for whether the output element can be populated
    bool imageOutputRowBoundaryValid = outputImageRow >= 0 && outputImageRow < imageHeight;
    bool imageOutputColBoundaryValid = outputImageCol >= 0 && outputImageCol < imageWidth;
    bool imageOutputBoundaryValid = imageOutputRowBoundaryValid && imageOutputColBoundaryValid;

    if(imageOutputBoundaryValid)
    {
        // Looping through the mask width in order to calculate the convolution
        float convolutionOutput = 0.0f;

        // Each thread between 0 - CONVOLUTION_TILE_WIDTH x CONVOLUTION_TILE_WIDTH will compute a convoltion.
        // tileRow and tileCol offset the 65x65 mask within the shared input tile.
        for(uint maskRow = 0; maskRow < CONVOLUTION_MASK_WIDTH; maskRow++)
        {
            for(uint maskCol = 0; maskCol < CONVOLUTION_MASK_WIDTH; maskCol++)
            {
                // Offset the mask position by this thread's output position
                // to access the corresponding 65x65 region in shared memory.
                convolutionOutput += inputTile[maskRow + tileRow][maskCol + tileCol] * mask[(maskRow * CONVOLUTION_MASK_WIDTH) + maskCol];
            }
        }

        // Update output image
        // Calculated by first finding the 1D image index.
        uint outputImageIdx = (outputImageRow * imageWidth) + outputImageCol;

        outputImage[outputImageIdx] = convolutionOutput;
    }
}