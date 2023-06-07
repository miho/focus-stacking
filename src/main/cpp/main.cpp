#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <cmath>

//// Equivalent function to Python's glob
std::vector<std::string> glob(const std::string& pattern) {
    std::vector<std::string> files;
    try {
        for (const auto &entry: std::filesystem::directory_iterator(pattern)) {
            if (!std::filesystem::is_directory(entry)) {
                files.push_back(entry.path().string());
            }
        }
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    return files;
}

// Calculate sharpness function
cv::Mat calculate_sharpness_old(const cv::Mat& image, int max_levels = 5) {
    cv::Mat grayscale;
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);

    std::vector<cv::Mat> pyramid;
    pyramid.push_back(grayscale);

    for (int i = 0; i < max_levels; i++) {
        cv::Mat next_level;
        cv::pyrDown(pyramid.back(), next_level);
        pyramid.push_back(next_level);
    }

    cv::Mat sharpness = cv::Mat::zeros(image.size(), CV_64F);

    for (const auto & level : pyramid) {
        cv::Mat laplacian, energy;
        cv::Laplacian(level, laplacian, CV_64F);
        energy = cv::abs(laplacian);
        cv::resize(energy, energy, image.size());
        sharpness += energy;
    }

    return sharpness;
}

cv::Mat calculate_sharpness(const cv::Mat& image, int max_levels = 3) {
    // Convert the image to grayscale
    cv::Mat grayscale;
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);

    // Construct the Laplacian pyramid
    std::vector<cv::Mat> pyramid{grayscale};
    for (int i = 0; i < max_levels; i++) {
        cv::Mat next_level;
        cv::pyrDown(pyramid.back(), next_level);
        pyramid.push_back(next_level);
    }

    // For each level in the pyramid, calculate the local energy
    std::vector<cv::Mat> energy_pyramid;
    for (auto& level : pyramid) {
        cv::Mat laplacian;
//        cv::Laplacian(level, laplacian, CV_64F);
        cv::Laplacian( level, laplacian, CV_64F, 3, /*scale*/0.25, /*delta*/0.0);
        cv::Mat energy;
        cv::convertScaleAbs(laplacian, energy);
        // Upscale to the original size
        cv::resize(energy, energy, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_LINEAR);
        energy_pyramid.push_back(energy);
    }

    // Sum up the energy images to get the combined sharpness map
    cv::Mat sharpness = cv::Mat::zeros(image.size(), CV_32F);
    for (auto& energy : energy_pyramid) {
        cv::Mat float_energy;
        energy.convertTo(float_energy, CV_32F);
        sharpness += float_energy;
    }

    return sharpness;
}

// Gamma correction function
cv::Mat gammaCorrection(const cv::Mat& image, double gamma) {
    cv::Mat lookup_table(1, 256, CV_8U);
    uchar* p = lookup_table.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);

    cv::Mat corrected_image;
    cv::LUT(image, lookup_table, corrected_image);

    return corrected_image;
}

cv::Mat blendImages(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks) {

    std::vector<cv::Mat> norm_masks;
    for (const auto& mask : masks) {
        cv::Mat float_mask;
        mask.convertTo(float_mask, CV_32F);
        cv::Mat mask3channel;
        cv::cvtColor(float_mask, mask3channel, cv::COLOR_GRAY2BGR);
        norm_masks.push_back(mask3channel / 255.0);
    }

    cv::Mat mask_sum = cv::Mat::zeros(images[0].size(), CV_32FC3);
    for (const auto& mask : norm_masks) {
        mask_sum += mask;
    }

    for (auto& mask : norm_masks) {
        cv::divide(mask, mask_sum, mask, 1, CV_32F);
    }

    cv::Mat blended = cv::Mat::zeros(images[0].size(), CV_32FC3);
    for (size_t i = 0; i < images.size(); i++) {
        cv::Mat float_image;
        images[i].convertTo(float_image, CV_32F);
        blended += float_image.mul(norm_masks[i]);
    }

    blended.convertTo(blended, CV_8U);

    return blended;
}


// Blend images function
cv::Mat blendImages_old(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks) {

    std::vector<cv::Mat> norm_masks;
    for (const auto& mask : masks) {
        cv::Mat float_mask;
        mask.convertTo(float_mask, CV_32F);
        norm_masks.push_back(float_mask / 255.0);
    }

    cv::Mat mask_sum = cv::Mat::zeros(images[0].size(), CV_32F);
    for (const auto& mask : norm_masks) {
        mask_sum += mask;
    }

    for (auto& mask : norm_masks) {
        cv::divide(mask, mask_sum, mask, 1, CV_32F);
    }

    cv::Mat blended = cv::Mat::zeros(images[0].size(), CV_32F);
    for (size_t i = 0; i < images.size(); i++) {
        cv::Mat float_image;
        images[i].convertTo(float_image, CV_32F);
        blended += float_image.mul(norm_masks[i]);
    }

    blended.convertTo(blended, CV_8U);

    return blended;
}

int main() {
    std::string src_directory = "C:/SP7-DATA/Users/miho/Downloads/focus-stacking-master/focus-stacking-master/example-images";

    // log folder
    std::cout << "Processing folder: " << src_directory << std::endl;

    // Create masks directory
    std::filesystem::create_directory("masks");

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> masks;

    std::vector<std::string> image_files = glob(src_directory);

    // Print the matched files
    for (const auto& file : image_files) {
        std::cout << file << std::endl;
    }

    for (size_t i = 0; i < image_files.size(); i++) {

        // log progress
        std::cout << "Processing image " << i + 1 << " of " << image_files.size() << std::endl;

        cv::Mat img = cv::imread(image_files[i]);
        images.push_back(img);

        cv::Mat sharpness_mask = calculate_sharpness(img);

        // Normalize
        double lower, upper;
        cv::minMaxIdx(sharpness_mask, &lower, &upper);
        sharpness_mask = (sharpness_mask - lower) / (upper - lower);

        // Convert to 8-bit (0-255)
        sharpness_mask.convertTo(sharpness_mask, CV_8U, 255);

        // sharpness_mask = gammaCorrection(sharpness_mask, 0);

        // mask min should be 1 instead of 0, max remains at 255
        sharpness_mask.setTo(1, sharpness_mask == 0);


        masks.push_back(sharpness_mask);

        // get file name without extension and parent path (Windows and Unix compatible
        std::string file_name = std::filesystem::path(image_files[i]).stem().string();

        // Save mask
        cv::imwrite("masks/"+file_name + ".png", sharpness_mask);
    }

    // Blend images
    cv::Mat result = blendImages(images, masks);

    // Save result
    cv::imwrite("result.png", result);

    return 0;
}
