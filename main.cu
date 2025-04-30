#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include "kernels.cuh"

namespace fs = std::filesystem;

std::vector<std::string> loadImagePaths(const std::string& folder) {
    std::vector<std::string> paths;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                paths.push_back(entry.path().string());
            }
        }
    }
    return paths;
}

int main() {
    std::vector<std::string> imagePaths = loadImagePaths("images");

    if (imagePaths.empty()) {
        std::cerr << "No images found in folder." << std::endl;
        return -1;
    }

    // Open CSV file
    std::ofstream logFile("output/benchmark.csv");
    logFile << "Image,Convolution(ms),MaxPooling(ms)\n";

    for (size_t i = 0; i < imagePaths.size(); ++i) {
        cv::Mat input = cv::imread(imagePaths[i], cv::IMREAD_GRAYSCALE);
        if (input.empty()) {
            std::cerr << "Failed to load: " << imagePaths[i] << std::endl;
            continue;
        }

        int width = input.cols;
        int height = input.rows;
        size_t imageSize = width * height * sizeof(unsigned char);

        // Allocate GPU memory
        unsigned char *d_input, *d_output;
        cudaMalloc(&d_input, imageSize);
        cudaMalloc(&d_output, imageSize);
        cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((width + 15) / 16, (height + 15) / 16);

        // === Convolution with Benchmark ===
        float convTime = 0;
        {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            conv2D_shared<<<gridSize, blockSize>>>(d_input, d_output, width, height);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&convTime, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        std::cout << "[Benchmark] Convolution time: " << convTime << " ms" << std::endl;

        // Copy and save convolution result
        cv::Mat convResult(height, width, CV_8UC1);
        cudaMemcpy(convResult.data, d_output, imageSize, cudaMemcpyDeviceToHost);
        std::string convPath = "output/conv_" + std::to_string(i) + ".png";
        cv::imwrite(convPath, convResult);

        // === Max Pooling with Benchmark ===
        int poolSize = 2;
        int pooledWidth = width / poolSize;
        int pooledHeight = height / poolSize;
        size_t pooledSize = pooledWidth * pooledHeight * sizeof(unsigned char);

        unsigned char* d_pooled;
        cudaMalloc(&d_pooled, pooledSize);

        dim3 poolGrid((pooledWidth + 15) / 16, (pooledHeight + 15) / 16);

        float poolTime = 0;
        {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            maxPool2D<<<poolGrid, blockSize>>>(d_output, d_pooled, width, height, poolSize);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&poolTime, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        std::cout << "[Benchmark] Max pooling time: " << poolTime << " ms" << std::endl;

        // Copy and save pooled result
        cv::Mat pooledOutput(pooledHeight, pooledWidth, CV_8UC1);
        cudaMemcpy(pooledOutput.data, d_pooled, pooledSize, cudaMemcpyDeviceToHost);
        std::string pooledPath = "output/pool_" + std::to_string(i) + ".png";
        cv::imwrite(pooledPath, pooledOutput);

        // Log benchmark to CSV
        logFile << fs::path(imagePaths[i]).filename().string() << "," << convTime << "," << poolTime << "\n";

        // Free GPU memory
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_pooled);

        std::cout << "Processed image " << i + 1 << "/" << imagePaths.size() << std::endl << std::endl;
    }

    logFile.close();
    std::cout << "All images processed and benchmark logged to output/benchmark.csv!" << std::endl;
    return 0;
}
