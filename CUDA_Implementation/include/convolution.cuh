/*
******************************************************************************
   File Name: convolution.cuh
   Description: Declares the CUDA convolution API and constants used for
   tiled convolution with the PSF mask.
******************************************************************************
*/

#ifndef CONVOLUTION_CUH
#define CONVOLUTION_CUH

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

// Kernel constants
const uint CONVOLUTION_MASK_WIDTH = 65;
const uint CONVOLUTION_MASK_RADIUS = CONVOLUTION_MASK_WIDTH / 2;

// Portion of input that was convolved
const uint CONVOLUTION_OUTPUT_TILE_WIDTH = 16;
// This is needed since at the first and last element, convolution needs the
// full mask. Since that will extend the input based on the radius of the mask
// twice since that extention happens at the first and last element.
const uint CONVOLUTION_INPUT_TILE_WIDTH = (CONVOLUTION_OUTPUT_TILE_WIDTH + CONVOLUTION_MASK_WIDTH - 1);

/*
******************************************************************************
* GLOBAL VARIABLES
******************************************************************************
*/

/*
******************************************************************************
* GLOBAL FUNCTION / KERNEL PROTOTYPES
******************************************************************************
*/

void convolveImage(
   // Inputs
   const float *deviceInputImage,
   const float *devicePsfMask,
   uint imageWidth,
   uint imageHeight,

   // Output
   float *deviceOutputImage
);

#endif // CONVOLUTION_CUH