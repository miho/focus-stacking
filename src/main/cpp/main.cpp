//
// Created by miho on 06.06.2023.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <thread>

const int PYRAMID_LEVELS = 5;

// Compute Laplacian pyramid
std::vector<cv::Mat> laplacianPyramid(const cv::Mat& img, int levels) {
    std::vector<cv::Mat> pyramid;
    cv::Mat currentImg = img;
    for (int i = 0; i < levels; ++i) {
        cv::Mat down, up;
        cv::pyrDown(currentImg, down);
        cv::pyrUp(down, up, currentImg.size());
        cv::Mat lap = currentImg - up;
        pyramid.push_back(lap);
        currentImg = down;
    }
    pyramid.push_back(currentImg);
    return pyramid;
}

std::vector<cv::Mat> blendPyramids(const std::vector<std::vector<cv::Mat>>& imgPyr, const std::vector<std::vector<cv::Mat>>& wPyr) {
    int n = imgPyr.size();
    int levels = imgPyr[0].size();

    std::vector<cv::Mat> blendedPyr(levels);
    for (int l = 0; l < levels; ++l) {
        cv::Mat& blendedImg = blendedPyr[l];

        // Initialize blendedImg with zeros, matching the size and type of the current image
        blendedImg = cv::Mat::zeros(imgPyr[0][l].size(), imgPyr[0][l].type());

        for (int i = 0; i < n; ++i) {
            cv::Mat img = imgPyr[i][l];
            cv::Mat weights = wPyr[i][l];

            // Ensure weights has the same number of channels as img
            if (weights.channels() == 1 && img.channels() == 3) {
                cv::Mat temp;
                std::vector<cv::Mat> channels = { weights, weights, weights };
                cv::merge(channels, temp);
                weights = temp;
            }

            // Now we can safely multiply img and weights and add to blendedImg
            cv::Mat weightedImg;
            cv::multiply(img, weights, weightedImg, 1, img.depth());
            cv::add(blendedImg, weightedImg, blendedImg, cv::noArray(), img.depth());
        }
    }
    return blendedPyr;
}


//std::vector<cv::Mat> blendPyramids(const std::vector<std::vector<cv::Mat>>& imgPyr, const std::vector<std::vector<cv::Mat>>& wPyr) {
//    int n = imgPyr.size();
//    int levels = imgPyr[0].size();
//
//    std::vector<cv::Mat> blendedPyr(levels);
//    for (int l = 0; l < levels; ++l) {
//        cv::Mat& blendedImg = blendedPyr[l];
//        for (int i = 0; i < n; ++i) {
//            cv::Mat img = imgPyr[i][l];
//            cv::Mat weights = wPyr[i][l];
//
//            // Convert weights to a 3-channel image if it's not already
//            if (weights.channels() == 1 && img.channels() == 3) {
//                cv::cvtColor(weights, weights, cv::COLOR_GRAY2BGR);
//            }
//
//            if (blendedImg.empty()) {
//                // If blendedImg is empty, this is the first image for this level
//                // Just multiply it by its weights and use that as the initial blendedImg
//                cv::multiply(img, weights, blendedImg, 1, img.depth());
//            } else {
//                // If blendedImg is not empty, add the weighted image to it
//                cv::Mat weightedImg;
//                cv::multiply(img, weights, weightedImg, 1, img.depth());
//                cv::add(blendedImg, weightedImg, blendedImg, cv::noArray(), img.depth());
//            }
//        }
//    }
//    return blendedPyr;
//}



// Collapse a Laplacian pyramid to get a single image
cv::Mat collapsePyramid(const std::vector<cv::Mat>& pyramid) {
    cv::Mat img = pyramid.back();
    for (int i = pyramid.size() - 2; i >= 0; --i) {
        cv::Mat up;
        cv::pyrUp(img, up, pyramid[i].size());
        img = up + pyramid[i];
    }
    return img;
}

std::vector<cv::Mat> gaussianPyramid(const cv::Mat& img, int levels) {
    std::vector<cv::Mat> pyramid(levels);
    pyramid[0] = img;
    for (int i = 1; i < levels; ++i) {
        cv::pyrDown(pyramid[i-1], pyramid[i], cv::Size(pyramid[i-1].cols/2, pyramid[i-1].rows/2));
    }
    return pyramid;
}



// Your helper function implementations for image pyramid method here...
// laplacianPyramid, blendPyramids, collapsePyramid

cv::Mat focusMeasure(const cv::Mat& img) {
    cv::Mat imgGray, imgLaplacian, fm;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(imgGray, imgLaplacian, CV_32F, 3, 1, 0, cv::BORDER_DEFAULT);

    cv::convertScaleAbs(imgLaplacian, fm);

//    cv::imshow("before mask", fm);
//    cv::waitKey(0);

    // clip dark values
    cv::threshold(fm, fm, 30, 0, cv::THRESH_TOZERO);

//    // show after mask
//    cv::imshow("after mask", fm);
//    cv::waitKey(0);

    cv::medianBlur(fm, fm, 3);  // Add median filter
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

void stackImagesPyramidThread(int startRow, int endRow, const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& focusMeasures, cv::Mat& stackedImage) {
    int n = images.size();
    int width = images[0].cols;
    int height = endRow - startRow;  // the height of the region this thread is responsible for

    std::cout << "!!! 1 !!!" << std::endl;

    // Create the Laplacian pyramids for the images and the Gaussian pyramids for the focus measures
    std::vector<std::vector<cv::Mat>> imgPyr(n), fmPyr(n);
    for (int i = 0; i < n; ++i) {
        imgPyr[i] = laplacianPyramid(images[i](cv::Rect(0, startRow, width, height)), PYRAMID_LEVELS);
        fmPyr[i] = gaussianPyramid(focusMeasures[i](cv::Rect(0, startRow, width, height)), PYRAMID_LEVELS);
    }

    std::cout << "!!! 2 !!!" << std::endl;

    // Blend the pyramids and collapse them back into a single image
    std::vector<cv::Mat> blendedPyr = blendPyramids(imgPyr, fmPyr);

    std::cout << "!!! 2b !!!" << std::endl;

    cv::Mat result = collapsePyramid(blendedPyr);

    std::cout << "!!! 3 !!!" << std::endl;

    // Write the result back to the correct rows of the stackedImage
    result.copyTo(stackedImage(cv::Rect(0, startRow, width, height)));

    std::cout << "!!! 4 !!!" << std::endl;
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./FocusStacking [image_directory] [method]\n";
        return -1;
    }

    std::string path = argv[1];
    std::string method = argv[2];

    std::vector<cv::Mat> images;
    for (const auto & entry : std::filesystem::directory_iterator(path)) {

        auto img = cv::imread(entry.path().string());

        if (img.empty()) {
            std::cout << "Failed to load image: " << entry.path() << "\n";
            continue;
        }

        // log the image name
        std::cout << entry.path() << "\n";

        images.push_back(img);
    }

    std::vector<cv::Mat> focusMeasures;
    for (const auto & img : images) {
        focusMeasures.push_back(focusMeasure(img));
    }

    cv::Mat stackedImage = cv::Mat::zeros(images[0].size(), images[0].type());

    // Determine the number of threads and rows per thread
    int numThreads = 1;//std::thread::hardware_concurrency();
    int rowsPerThread = stackedImage.rows / numThreads;

    // Launch threads for stacking
    std::vector<std::thread> threads;
    if (method == "simple") {
        for (int i = 0; i < numThreads; ++i) {
            int startRow = i * rowsPerThread;
            // Ensure the last thread processes the remaining rows if the number of rows isn't divisible evenly
            int endRow = (i == numThreads - 1) ? stackedImage.rows : startRow + rowsPerThread;
            threads.push_back(std::thread(stackImagesThread, startRow, endRow, std::ref(images), std::ref(focusMeasures), std::ref(stackedImage)));
        }
    } else if (method == "pyramid") {
        for (int i = 0; i < numThreads; ++i) {
            int startRow = i * rowsPerThread;
            int endRow = (i == numThreads - 1) ? stackedImage.rows : startRow + rowsPerThread;
            threads.push_back(std::thread(stackImagesPyramidThread, startRow, endRow, std::ref(images), std::ref(focusMeasures), std::ref(stackedImage)));
        }
    } else {
        std::cout << "Invalid method. Choose 'simple' or 'pyramid'.\n";
        return -1;
    }

    // Wait for all threads to finish
    for (std::thread & t : threads) {
        t.join();
    }

    cv::imshow("Stacked Image", stackedImage);
    cv::waitKey(0);

    return 0;
}
