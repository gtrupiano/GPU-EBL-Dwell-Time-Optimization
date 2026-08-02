/*
**********************************************************************
	File Name: error_calculations.cu
	Description:
**********************************************************************
*/

/*
 **********************************************************************
 * INCLUDES
 **********************************************************************
*/

#include "error_calculations.cuh"

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
 * LOCAL FUNCTION / KERNEL PROTOTYPES
 **********************************************************************
*/

// Functions
static void calculateErrorMatrix(
    const float *deviceTargetLayout,
    const float *deviceDepositedEnergy,
    uint imageWidth,
    uint imageHeight,
    float *deviceErrorMatrix,
    float *deviceSquaredErrorSum
);

static float calculateMSE(
   const float *deviceSquaredErrorSum,
   uint imageWidth,
   uint imageHeight
);


// Kernels
static __global__ void errorMatrixCalculationKernel(
   const float *icLayout,
   const float *depositedEnergy,
   uint imageWidth,
   uint imageHeight,
   float *errorMatrix,
   float *squaredErrorSum
);

/*
 **********************************************************************
 * GLOBAL FUNCTIONS
 **********************************************************************
*/

/**************************************************
 * Function: calculateError
 * Description: 
**************************************************/

float calculateError(
	const float *deviceTargetLayout,
	const float *deviceDepositedEnergy,
	uint imageWidth,
	uint imageHeight,
	float *deviceErrorMatrix,
	float *deviceSquaredErrorSum
)
{
	calculateErrorMatrix(
		deviceTargetLayout,
        deviceDepositedEnergy,
        imageWidth,
        imageHeight,
        deviceErrorMatrix,
        deviceSquaredErrorSum
	);

	float mse = calculateMSE(
		deviceSquaredErrorSum,
		imageWidth,
		imageHeight
    );

	return mse;
}


/*
 **********************************************************************
 * LOCAL FUNCTIONS
 **********************************************************************
*/

/**************************************************
 * Function: calculateErrorMatrix
 * Description: 
**************************************************/

static void calculateErrorMatrix(
	const float *deviceTargetLayout,
	const float *deviceDepositedEnergy,
	uint imageWidth,
	uint imageHeight,
	float *deviceErrorMatrix,
	float *deviceSquaredErrorSum
)
{
    // Threads per block value for each dimention used
    dim3 blockSize(ERROR_BLOCK_WIDTH, ERROR_BLOCK_HEIGHT, 1);

	// Grid size
    // There needs to be enough threads for each row,col of the matrix.
    // So by dividing length of matrix / threadsPerBlock, it should yield number of blocks.
    // Rounding up is needed (integer based rounding
    uint blocksPerDimX = (imageWidth + ERROR_BLOCK_WIDTH - 1) / ERROR_BLOCK_WIDTH;
    uint blocksPerDimY = (imageHeight + ERROR_BLOCK_HEIGHT - 1) / ERROR_BLOCK_HEIGHT;
    dim3 numberOfBlocks(blocksPerDimX, blocksPerDimY, 1);

	// Settin to 0 in order to verify squared error sum starts as 0
    cudaMemset(deviceSquaredErrorSum, 0, sizeof(float));

	// Launching kernel
    errorMatrixCalculationKernel<<<numberOfBlocks, blockSize>>>(
        deviceTargetLayout,
        deviceDepositedEnergy,
        imageWidth,
        imageHeight,
        deviceErrorMatrix,
        deviceSquaredErrorSum
    );
}


/**************************************************
 * Function: calculateMSE
 * Description: 
**************************************************/

static float calculateMSE(
    const float *deviceSquaredErrorSum,
    uint imageWidth,
    uint imageHeight
)
{
    float squaredErrorSum = 0.0f;

    cudaMemcpy(
        &squaredErrorSum,
        deviceSquaredErrorSum,
        sizeof(float),
        cudaMemcpyDeviceToHost
    );

    uint numberOfPixels = imageWidth * imageHeight;

    float mse = squaredErrorSum / float(numberOfPixels);

	return mse;
}


/*
 **********************************************************************
 * KERNELS
 **********************************************************************
*/

/**************************************************
 * Kernel: errorMatrixCalculationKernel
 * Description: 
**************************************************/

static __global__ void errorMatrixCalculationKernel(
   const float *targetLayout,
   const float *depositedEnergy,
   uint imageWidth,
   uint imageHeight,
   float *errorMatrix,
   float *squaredErrorSum
)
{
	// Each thread stores its squared error in shared memory but done in 1D for easy reduction
	__shared__ float sharedSquaredErrors[ERROR_BLOCK_SIZE];

	uint ty = threadIdx.y;
	uint tx = threadIdx.x;
	uint rowIdx = (blockIdx.y * blockDim.y) + ty;
	uint colIdx = (blockIdx.x * blockDim.x) + tx;

	// 1D representation for global and shared memory
	uint globalPixelIdx1D = (rowIdx * imageWidth) + colIdx;
	uint sharedPixelIdx1D = (ty * ERROR_BLOCK_WIDTH) + tx;

	// Defaulting shared memory to 0
	sharedSquaredErrors[sharedPixelIdx1D] = 0.0f;

	// Boundary checking
	if(rowIdx < imageHeight && colIdx < imageWidth)
	{		
		// Calculating error between expected vs actual
		float error = targetLayout[globalPixelIdx1D] - depositedEnergy[globalPixelIdx1D];

		// Updating error matrix with error
		errorMatrix[globalPixelIdx1D] = error;

		// Updating shared memeory with squared error for later MSE calculation
		sharedSquaredErrors[sharedPixelIdx1D] = error * error;
	}

	// Waiting for all threads to calculate standard and squared error
	__syncthreads();


	// Reduction via progressively smaller strides.
	// Each block has its own sharedSquaredErrors array.
	// Active threads repeatedly sum multiple values separated by the current stride
	// until the block's total squared error is stored in sharedSquaredErrors[0].
	for(uint stride = ERROR_BLOCK_SIZE / 2; stride > 0; stride /= 2)
    {
		// Making sure only threads associated with data are being used for calculation
        if (sharedPixelIdx1D < stride)
        {
            sharedSquaredErrors[sharedPixelIdx1D] += sharedSquaredErrors[sharedPixelIdx1D + stride];
        }
		
		// Waiting for all threads to do their summation calculations before proceeding
        __syncthreads();
    }

	// Updating the output object with all of the partial sums from each block
    if (sharedPixelIdx1D == 0)
    {
		// With sharedSquaredErrors[0] holding the squared error sum, 
		// the host must take the squaredErrorSum and calculate the mse
		// by doing the following calculation:
		// 		squaredErrorSum / (imageWidth * imageHeight)
        atomicAdd(squaredErrorSum, sharedSquaredErrors[0]);
    }
}