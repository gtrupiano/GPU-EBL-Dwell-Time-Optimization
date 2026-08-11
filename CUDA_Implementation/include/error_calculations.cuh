/*
 ******************************************************************************
   File Name  : error_calculations.cuh
   Description:
 ******************************************************************************
*/
  
#ifndef ERROR_CALCULATIONS_CUH
#define ERROR_CALCULATIONS_CUH
  
/*
 ******************************************************************************
 * INCLUDES
 ******************************************************************************
*/

#include <cuda_runtime.h>

/*
 ******************************************************************************
 * DEFINES, CONSTANTS, ENUMS, STRUCTS
 ******************************************************************************
*/

const uint ERROR_BLOCK_WIDTH = 16;
const uint ERROR_BLOCK_HEIGHT = 16;
const uint ERROR_BLOCK_SIZE = ERROR_BLOCK_WIDTH * ERROR_BLOCK_HEIGHT;

/*
 ******************************************************************************
 * GLOBAL VARIABLES
 ******************************************************************************
*/

/*
 ******************************************************************************
 * GLOBAL FUNCTION PROTOTYPES
 ******************************************************************************
*/

float calculateError(
	// Inputs
	const float *deviceTargetLayout,
	const float *deviceDepositedEnergy,
	uint imageWidth,
	uint imageHeight,

	// Intermediate
	float *deviceSquaredErrorSum,

	// Output
	float *deviceErrorMatrix
);

#endif // ERROR_CALCULATIONS_CUH