//
// Created by miho on 06.06.2023.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <thread>

cv::Mat focusMeasure(const cv::Mat& img) {
    cv::Mat imgGray, imgLaplacian, fm;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(imgGray, imgLaplacian, CV_32F, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(imgLaplacian, fm);

    // Add median filter
    cv::medianBlur(fm, fm, 21);

    cv::imshow("GRAY Image", fm);
    cv::waitKey(0);

    return fm;
}

void stackImagesThread(int startRow, int endRow, const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& focusMeasures, cv::Mat& stackedImage) {
    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < stackedImage.cols; ++x) {
            int bestFocusIdx = 0;
            for (int i = 0; i < images.size(); ++i) {
                if (focusMeasures[i].at<uchar>(y, x) > focusMeasures[bestFocusIdx].at<uchar>(y, x)) {
                    bestFocusIdx = i;
                }
            }
            stackedImage.at<cv::Vec3b>(y, x) = images[bestFocusIdx].at<cv::Vec3b>(y, x);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./FocusStacking [image_directory]\n";
        return -1;
    }

    std::string path = argv[1];
    std::vector<cv::Mat> images;
    for (const auto & entry : std::filesystem::directory_iterator(path)) {
        // log the path to the console
        auto fPath = entry.path().string();
        std::cout << fPath << std::endl;
        images.push_back(cv::imread(fPath));
    }

    std::vector<cv::Mat> focusMeasures;
    for (const auto & img : images) {
        focusMeasures.push_back(focusMeasure(img));
    }

    cv::Mat stackedImage = cv::Mat::zeros(images[0].size(), images[0].type());

    // Determine the number of threads and rows per thread
    int numThreads = std::thread::hardware_concurrency();
    int rowsPerThread = stackedImage.rows / numThreads;

    // Launch threads for stacking
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i == numThreads - 1) ? stackedImage.rows : startRow + rowsPerThread;  // make sure the last thread processes all remaining rows
        threads.push_back(std::thread(stackImagesThread, startRow, endRow, std::ref(images), std::ref(focusMeasures), std::ref(stackedImage)));
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    cv::imshow("Stacked Image", stackedImage);
    cv::waitKey(0);

    return 0;
}
