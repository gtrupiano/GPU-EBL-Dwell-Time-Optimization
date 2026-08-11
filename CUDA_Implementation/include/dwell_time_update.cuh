/*
 ******************************************************************************
    File Name  : dwell_time_update.cuh
    Description: Declares the API and constants used to calculate and apply
    dwell-time corrections to the dwell-time map.
 ******************************************************************************
*/
  
#ifndef DWELL_TIME_UPDATE_CUH
#define DWELL_TIME_UPDATE_CUH
  
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

const uint DWELL_BLOCK_WIDTH = 16;
const uint DWELL_BLOCK_HEIGHT = 16;

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
);

#endif // DWELL_TIME_UPDATE_CUH