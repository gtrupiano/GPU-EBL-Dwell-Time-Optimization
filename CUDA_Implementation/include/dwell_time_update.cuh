/*
 ******************************************************************************
    File Name  : dwell_time_update.cuh
    Description:
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

#define DWELL_BLOCK_WIDTH 16
#define DWELL_BLOCK_HEIGHT 16

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
    float *deviceDwellTimeMap,
    const float *deviceErrorMatrix,
    const float *devicePsfMask,
    float *deviceDwellTimeCorrection,
    uint imageWidth,
    uint imageHeight,
    float learningRate,
    float maxDwellTime
);

#endif // DWELL_TIME_UPDATE_CUH