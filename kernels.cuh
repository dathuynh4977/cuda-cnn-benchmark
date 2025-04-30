#pragma once

__global__ void conv2D_shared(unsigned char* input, unsigned char* output, int width, int height);

__global__ void maxPool2D(unsigned char* input, unsigned char* output, int width, int height, int poolSize);
