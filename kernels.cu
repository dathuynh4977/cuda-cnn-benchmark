#include "kernels.cuh"

__constant__ int filter[3][3] = {
    { 0, -1,  0 },
    { -1, 5, -1 },
    { 0, -1,  0 }
};

// =======================
// Basic Global Memory Version
// =======================
__global__ void conv2D(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // col
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int sum = 0;
        for (int fy = -1; fy <= 1; ++fy) {
            for (int fx = -1; fx <= 1; ++fx) {
                int pixel = input[(y + fy) * width + (x + fx)];
                sum += pixel * filter[fy + 1][fx + 1];
            }
        }
        sum = min(max(sum, 0), 255);
        output[y * width + x] = (unsigned char)sum;
    }
}

// =======================
// Shared Memory Optimized Version
// =======================
__global__ void conv2D_shared(unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ unsigned char tile[18][18];  // 16x16 block + 1-pixel halo on each side

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * 16 + tx;
    int y = blockIdx.y * 16 + ty;

    // Load center pixel
    if (x < width && y < height)
        tile[ty + 1][tx + 1] = input[y * width + x];

    // Load halos (edges and corners)
    if (tx == 0 && x > 0)
        tile[ty + 1][0] = input[y * width + (x - 1)];
    if (tx == 15 && x < width - 1)
        tile[ty + 1][17] = input[y * width + (x + 1)];
    if (ty == 0 && y > 0)
        tile[0][tx + 1] = input[(y - 1) * width + x];
    if (ty == 15 && y < height - 1)
        tile[17][tx + 1] = input[(y + 1) * width + x];

    // Corners
    if (tx == 0 && ty == 0 && x > 0 && y > 0)
        tile[0][0] = input[(y - 1) * width + (x - 1)];
    if (tx == 15 && ty == 0 && x < width - 1 && y > 0)
        tile[0][17] = input[(y - 1) * width + (x + 1)];
    if (tx == 0 && ty == 15 && x > 0 && y < height - 1)
        tile[17][0] = input[(y + 1) * width + (x - 1)];
    if (tx == 15 && ty == 15 && x < width - 1 && y < height - 1)
        tile[17][17] = input[(y + 1) * width + (x + 1)];

    __syncthreads();

    // Perform convolution
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1 && tx < 16 && ty < 16) {
        int sum = 0;
        for (int fy = -1; fy <= 1; ++fy)
            for (int fx = -1; fx <= 1; ++fx)
                sum += tile[ty + 1 + fy][tx + 1 + fx] * filter[fy + 1][fx + 1];

        sum = min(max(sum, 0), 255);
        output[y * width + x] = (unsigned char)sum;
    }
}

// =======================
// Max Pooling Kernel
// =======================
__global__ void maxPool2D(unsigned char* input, unsigned char* output, int width, int height, int poolSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int outWidth = width / poolSize;
    int outHeight = height / poolSize;

    if (x < outWidth && y < outHeight) {
        unsigned char maxVal = 0;
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int in_x = x * poolSize + j;
                int in_y = y * poolSize + i;
                if (in_x < width && in_y < height) {
                    unsigned char val = input[in_y * width + in_x];
                    if (val > maxVal) maxVal = val;
                }
            }
        }
        output[y * outWidth + x] = maxVal;
    }
}
