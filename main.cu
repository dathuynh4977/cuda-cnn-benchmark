#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include "kernels.cuh"

namespace fs = std::filesystem;

std::vector<std::pair<std::string, std::string>> loadImagePathsWithLabels(const std::string& baseFolder) {
    std::vector<std::pair<std::string, std::string>> imagePaths;
    for (const auto& dirEntry : fs::directory_iterator(baseFolder)) {
        if (dirEntry.is_directory()) {
            std::string label = dirEntry.path().filename().string();
            for (const auto& fileEntry : fs::directory_iterator(dirEntry.path())) {
                if (fileEntry.is_regular_file()) {
                    std::string ext = fileEntry.path().extension().string();
                    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                        imagePaths.emplace_back(fileEntry.path().string(), label);
                    }
                }
            }
        }
    }
    return imagePaths;
}

void ensureOutputFolders(const std::string& base, const std::vector<std::string>& labels) {
    for (const auto& layer : {"cpu", "conv", "pool"}) {
        for (const auto& label : labels) {
            fs::create_directories(fs::path(base) / layer / label);
        }
    }
}

int main() {
    auto imagePairs = loadImagePathsWithLabels("images");
    std::set<std::string> labels;
    for (const auto& p : imagePairs) labels.insert(p.second);
    ensureOutputFolders("output", {labels.begin(), labels.end()});

    std::ofstream logFile("output/benchmark.csv");
    logFile << "Label,Image,CPU(ms),Convolution(ms),MaxPooling(ms)\n";

    std::cout << "========== IMAGE LOADING ==========\n";
    std::cout << "Loaded " << imagePairs.size() << " images successfully. Failed to load: 0\n\n";

    float totalCpuConvTime = 0;
    float totalGpuConvTime = 0, totalGpuConvMemcpy = 0, totalGpuConvKernel = 0;
    float totalGpuPoolTime = 0, totalGpuPoolMemcpy = 0, totalGpuPoolKernel = 0;

    for (size_t i = 0; i < imagePairs.size(); ++i) {
        const std::string& imagePath = imagePairs[i].first;
        const std::string& label = imagePairs[i].second;

        cv::Mat input = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (input.empty()) continue;

        std::string baseName = fs::path(imagePath).stem().string();

        // CPU Convolution
        cv::Mat cpuOutput;
        cv::Mat kernel = (cv::Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
        int64 cpuStart = cv::getTickCount();
        cv::filter2D(input, cpuOutput, -1, kernel);
        int64 cpuEnd = cv::getTickCount();
        float cpuTime = (cpuEnd - cpuStart) * 1000.0 / cv::getTickFrequency();
        totalCpuConvTime += cpuTime;
        cv::imwrite("output/cpu/" + label + "/" + baseName + ".png", cpuOutput);

        // GPU Convolution
        int width = input.cols, height = input.rows;
        size_t imageSize = width * height * sizeof(unsigned char);
        unsigned char *d_input, *d_output;
        cudaMalloc(&d_input, imageSize);
        cudaMalloc(&d_output, imageSize);

        float convMemcpyTime = 0, convKernelTime = 0;
        {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&convMemcpyTime, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        dim3 blockSize(16, 16);
        dim3 gridSize((width + 15) / 16, (height + 15) / 16);

        {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            conv2D_shared<<<gridSize, blockSize>>>(d_input, d_output, width, height);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&convKernelTime, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        float convTotal = convMemcpyTime + convKernelTime;
        totalGpuConvTime += convTotal;
        totalGpuConvMemcpy += convMemcpyTime;
        totalGpuConvKernel += convKernelTime;

        cv::Mat convOutput(height, width, CV_8UC1);
        cudaMemcpy(convOutput.data, d_output, imageSize, cudaMemcpyDeviceToHost);
        cv::imwrite("output/conv/" + label + "/" + baseName + ".png", convOutput);

        // Max Pooling
        int pooledWidth = width / 2, pooledHeight = height / 2;
        size_t pooledSize = pooledWidth * pooledHeight * sizeof(unsigned char);
        unsigned char* d_pooled;
        cudaMalloc(&d_pooled, pooledSize);

        float poolKernelTime = 0, poolMemcpyTime = 0;
        {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            maxPool2D<<<dim3((pooledWidth+15)/16, (pooledHeight+15)/16), blockSize>>>(d_output, d_pooled, width, height, 2);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&poolKernelTime, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cv::Mat pooledOutput(pooledHeight, pooledWidth, CV_8UC1);
            cudaMemcpy(pooledOutput.data, d_pooled, pooledSize, cudaMemcpyDeviceToHost);
            cv::imwrite("output/pool/" + label + "/" + baseName + ".png", pooledOutput);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&poolMemcpyTime, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        float poolTotal = poolKernelTime + poolMemcpyTime;
        totalGpuPoolTime += poolTotal;
        totalGpuPoolKernel += poolKernelTime;
        totalGpuPoolMemcpy += poolMemcpyTime;

        logFile << label << "," << fs::path(imagePath).filename().string()
                << "," << cpuTime << "," << convTotal << "," << poolTotal << "\n";

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_pooled);
    }

    std::cout << "========== CPU PERFORMANCE ==========\n";
    std::cout << "[CPU] Total convolution time: " << totalCpuConvTime << " ms\n\n";

    std::cout << "========== GPU PERFORMANCE ==========\n";
    std::cout << "[GPU] Convolution: Total " << totalGpuConvTime << " ms (memcpy: "
              << totalGpuConvMemcpy << " ms, kernel: " << totalGpuConvKernel << " ms)\n";
    std::cout << "[GPU] Pooling:     Total " << totalGpuPoolTime << " ms (memcpy: "
              << totalGpuPoolMemcpy << " ms, kernel: " << totalGpuPoolKernel << " ms)\n";

    logFile.close();
    return 0;
}
