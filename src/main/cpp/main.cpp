#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <cmath>
#include <args.hxx>

#include "main/include/glob/glob.hpp"

namespace fs = std::filesystem;

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

    // log blending
    std::cout << "blending images:" << std::endl;

    std::vector<cv::Mat> norm_masks;
    std::cout << " -> converting masks" << std::endl;
    for (const auto& mask : masks) {
        cv::Mat float_mask;
        mask.convertTo(float_mask, CV_32F);
        cv::Mat mask3channel;
        cv::cvtColor(float_mask, mask3channel, cv::COLOR_GRAY2BGR);
        norm_masks.push_back(mask3channel / 255.0);
    }

    cv::Mat mask_sum = cv::Mat::zeros(images[0].size(), CV_32FC3);

    std::cout << " -> summing masks" << std::endl;
    for (const auto& mask : norm_masks) {
        mask_sum += mask;
    }

    std::cout << " -> dividing masks" << std::endl;
    for (auto& mask : norm_masks) {
        cv::divide(mask, mask_sum, mask, 1, CV_32F);
    }

    std::cout << " -> blending images" << std::endl;
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

#include <args.hxx>
#include <iostream>
#include <string>

struct ProgramOptions
{
    bool version = false;
    std::string input_folder;
    std::string output_folder;
    std::string mask_folder;
    bool verbose = false;
    int kernelSize = 3;
    double threshold = 0.0;
    double scale = 1.0;
    double gamma = 1.0;
    int numLayers = 2;
};

ProgramOptions parse_and_check_args(int argc, char **argv)
{
    ProgramOptions options;

    args::ArgumentParser parser("This is a focus stacking program.", "Example: focus_stacker -i input_folder -o output_folder");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag version(parser, "version", "Display the version number", {'v', "version"});
    args::ValueFlag<std::string> input(parser, "input", "The input files/folder", {'i', "input"});
    args::ValueFlag<std::string> output(parser, "output", "The output folder", {'o', "output"});
    args::ValueFlag<std::string> mask(parser, "masks", "The mask files/folder", {'m', "mask"});
    args::Flag verbose(parser, "verbose", "Verbose output", {'V', "verbose"});
    args::ValueFlag<int> kernelSize(parser, "kernel-size", "The Laplace kernel size", {'k', "kernel-size"}, 3);
    args::ValueFlag<double> threshold(parser, "threshold", "The Laplace threshold", {'t', "threshold"}, 0.0);
    args::ValueFlag<double> scale(parser, "scale", "The Laplace scale", {'s', "scale"}, 1.0);
    args::ValueFlag<double> gamma(parser, "gamma", "The Laplace gamma", {'g', "gamma"}, 1.0);
    args::ValueFlag<int> numLayers(parser, "num-layers", "The number of layers for Laplace pyramid", {'l', "num-layers"}, 2);

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help& h)
    {
        // print help
        std::cout << parser;
        exit(0);
    }
    catch (args::ParseError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        exit(1);
    }
    catch (args::ValidationError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        exit(1);
    }

    if(version) {
        options.version = true;
    }

    if(input) {
        options.input_folder = args::get(input);

        // check it's a folder
        if(!fs::is_directory(options.input_folder)) {
            std::cerr << "Input must be a folder" << std::endl;
            std::cerr << parser;
            exit(1);
        }

    } else {
        std::cerr << "Input is required" << std::endl;
        std::cerr << parser;
        exit(1);
    }

    if(output) {
        options.output_folder = args::get(output);

        // check it's a folder
        if(!fs::is_directory(options.output_folder)) {
            // if is file, show error
            if(fs::is_regular_file(options.output_folder)) {
                std::cerr << "ERROR: Output must be a folder" << std::endl;
                std::cerr << parser;
                exit(1);
            }
            else if(options.output_folder == options.input_folder) {
                std::cerr << "ERROR: Output must be different from input" << std::endl;
                std::cerr << parser;
                exit(1);
            }
            else {
                // if not, create it
                fs::create_directory(options.output_folder);
            }
        }

    } else {
        std::cerr << "ERROR: Output is required" << std::endl;
        std::cerr << parser;
        exit(1);
    }

    if(mask) {
        options.mask_folder = args::get(mask);

        // check it's a folder
        if(!fs::is_directory(options.mask_folder)) {
            std::cerr << "ERROR: Masks must be a folder" << std::endl;
            std::cerr << parser;
            exit(1);
        }
    }

    if(verbose) {
        options.verbose = true;
    }

    if(kernelSize) {
        options.kernelSize = args::get(kernelSize);

        if(options.kernelSize < 1) {
            std::cerr << "ERROR: Kernel size must be greater than 0" << std::endl;
            std::cerr << parser;
            exit(1);
        } else if(options.kernelSize % 2 == 0) {
            std::cerr << "ERROR: Kernel size must be odd" << std::endl;
            std::cerr << parser;
            exit(1);
        }
    }

    if(threshold) {
        options.threshold = args::get(threshold);

        // warn, currently not used
        std::cout << "WARNING: threshold is currently not used" << std::endl;
    }

    if(scale) {
        options.scale = args::get(scale);

        // warn, currently not used
        std::cout << "WARNING: scale is currently not used" << std::endl;
    }

    if(gamma) {
        options.gamma = args::get(gamma);
    }

    if(numLayers) {
        options.numLayers = args::get(numLayers);

        if(options.numLayers < 1) {
            std::cerr << "ERROR: Number of layers must be greater than 0" << std::endl;
            std::cerr << parser;
            exit(1);
        } else if(options.numLayers > 10) {
            // wow, I hope you know what you are doing
            std::cout << "WARNING: Wow, I hope you know what you are doing ;)" << std::endl;
        }
    }

    return options;
}

int main(int argc, char **argv)
{

    ProgramOptions options = parse_and_check_args(argc, argv);

    std::string src_directory = options.input_folder;

    // log folder
    std::cout << "Processing folder: " << src_directory << std::endl;

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> masks;

    // glob pattern for matching jpg and png
    std::string pattern = src_directory + "/*.{[jJ][pP][gG],[pP][nN][gG]}";

    std::vector<std::string> supported_imgs = {".jpg", ".png"};

    // get all files in the folder and check whether they are images (contained in supported imgs), use to_lower to make it case insensitive
    std::vector<std::string> image_files;
    for (const auto& entry : std::filesystem::directory_iterator(src_directory)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
        if (std::find(supported_imgs.begin(), supported_imgs.end(), ext) != supported_imgs.end()) {
            std::string path_str = entry.path().string();
            image_files.push_back(path_str);
        }
    }

    for (size_t i = 0; i < image_files.size(); i++) {
        std::cout << "reading image " << i + 1 << " of " << image_files.size() << std::endl;

        cv::Mat img = cv::imread(image_files[i]);

        if (img.empty()) {
            std::cout << "WARNING: Could not read image " << image_files[i] << std::endl;
        } else {
            images.push_back(img);
        }

    }

    for (size_t i = 0; i < images.size(); i++) {

        auto img = images[i];

        // log progress
        std::cout << "analyzing sharpness of image " << (i + 1) << " of " << image_files.size() << std::endl;

        cv::Mat sharpness_mask = calculate_sharpness(img);

        // Normalize
        double lower, upper;
        cv::minMaxIdx(sharpness_mask, &lower, &upper);
        sharpness_mask = (sharpness_mask - lower) / (upper - lower);

        // Convert to 8-bit (0-255)
        sharpness_mask.convertTo(sharpness_mask, CV_8U, 255);

        sharpness_mask = gammaCorrection(sharpness_mask, options.gamma);

        // mask min should be 1 instead of 0, max remains at 255
        sharpness_mask.setTo(1, sharpness_mask == 0);

        masks.push_back(sharpness_mask);

        // get file name without extension and parent path (Windows and Unix compatible
        std::string file_name = std::filesystem::path(image_files[i]).stem().string();

        // Save mask
        cv::imwrite("masks/"+file_name + ".png", sharpness_mask);
    }

    if (images.empty()) {
        std::cerr << "ERROR: No images found" << std::endl;
        exit(1);
    }

    // Blend images
    cv::Mat result = blendImages(images, masks);

    // Save result
    cv::imwrite(options.output_folder+"/stacked.png", result);

    return 0;
}
