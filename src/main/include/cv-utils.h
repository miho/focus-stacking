#pragma once

#include <sstream>
#include <fstream>
#include <filesystem>

#include<opencv2/opencv.hpp>
//#include "nlohmann/json.hpp"

#include "log-utils.h"

void fill_binary_image(const cv::Mat& binary_input, cv::Mat& dst, cv::Point flood_location = cv::Point(0,0));
std::string type2str(int type);

// from opencv docs
// see https://docs.opencv.org/4.7.0/dd/d3d/tutorial_gpu_basics_similarity.html
cv::Scalar computeMSSIM( const cv::Mat& i1, const cv::Mat& i2);
cv::Scalar computeMSSIM( const cv::Mat& i1, const cv::Mat& i2, cv::Mat& ssim_map_dst);

void detectDifference(cv::Mat &img1, cv::Mat &img2, cv::Mat &img3,
                      int num_tiles_x, int num_tiles_y,
                      std::vector<int> &tile_result);

std::tuple<std::vector<std::string>, std::vector<std::string>> read_images_to_stack(
        const std::string &input_folder_path, std::vector<cv::Mat> &images);

void compute_diff(cv::Mat img_1, cv::Mat img_2, cv::Mat& diff_img, double img_scale, int kernel_size, std::vector<int>& tiles, int num_tiles_x, int num_tiles_y);

void align_image(const cv::Mat &ref_img, const cv::Mat &img, const cv::Mat &img_orig, cv::Mat &img_aligned);

std::tuple<cv::Mat,std::vector<cv::Mat>> stack_images(
        const std::vector<cv::Mat> &images,
        const float grayscale_factor=0.25f);

cv::Mat calculate_sharpness(const cv::Mat& image, int max_levels = 3, int kernel_size = 3, double scale = 0.25, double delta = 0.0);
cv::Mat gammaCorrection(const cv::Mat& image, double gamma);
cv::Mat blendImages(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& masks);

//struct app_config {
//    public:
//    uint32_t num_threads;
//    std::string host;
//    uint32_t port;
//
//    static void from_json(app_config& cfg, nlohmann::json const &j) {
//
//        if(!j.contains("number-of-threads")) {
//            LOG_ERROR("Cannot find 'number-of-threads' in config file");
//        }
//        j.at("number-of-threads").get_to(cfg.num_threads);
//
//        if(!j.contains("host")) {
//            LOG_ERROR("Cannot find 'host' in config file");
//        }
//        j.at("host").get_to(cfg.host);
//
//        if(!j.contains("port")) {
//            LOG_ERROR("Cannot find 'port' in config file");
//        }
//        j.at("port").get_to(cfg.port);
//
//        std::stringstream ss;
//        ss << j.dump(2) << std::endl;
//        LOG_INFO("using config: \n" + ss.str());
//    }
//
//    static void from_file(app_config& cfg, std::filesystem::path config_file) {
//        // read config
//        std::ifstream is(config_file);
//        nlohmann::json j;
//        is >> j;
//        from_json(cfg, j);
//    }
//
//    static void to_json(nlohmann::json& j, const app_config& cfg) {
//        j = nlohmann::json{{"number-of-threads", cfg.num_threads}, {"port", cfg.port}};
//    }
//};




