#include "cv-utils.h"
#include "app-utils.h"

#include<opencv2/opencv.hpp>

void fill_binary_image(const cv::Mat& binary_input, cv::Mat& dst, cv::Point flood_location) {

    // Flood-fill background with white from flood_location
    cv::Mat flood_filled_img = binary_input.clone();
    cv::floodFill(flood_filled_img, flood_location, cv::Scalar(255));

    // Invert the flood-filled image
    cv::Mat flood_filled_img_inverted;
    bitwise_not(flood_filled_img, flood_filled_img_inverted);

    // combine the two images to get the foreground.
    cv::bitwise_or(binary_input, flood_filled_img_inverted, dst);
}

std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

// from opencv docs
cv::Scalar computeMSSIM( const cv::Mat& i1, const cv::Mat& i2) {

    cv::Mat ssim_map_dst;

    return computeMSSIM(i1, i2, ssim_map_dst);
}

// from opencv docs
cv::Scalar computeMSSIM( const cv::Mat& i1, const cv::Mat& i2, cv::Mat& ssim_map_dst) {
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;
    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    cv::Mat I2_2   = I2.mul(I2);        // I2^2
    cv::Mat I1_2   = I1.mul(I1);        // I1^2
    cv::Mat I1_I2  = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
    cv::Mat mu1_2   =   mu1.mul(mu1);
    cv::Mat mu2_2   =   mu2.mul(mu2);
    cv::Mat mu1_mu2 =   mu1.mul(mu2);
    cv::Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    // write t1 and t3 to files
//    cv::imwrite("t1.png", t1);
//    cv::imwrite("t2.png", t2);

    //cv::Mat ssim_map;
    divide(t3, t1, ssim_map_dst);      // ssim_map =  t3./t1;
    cv::Scalar mssim = mean( ssim_map_dst ); // mssim = average of ssim map

    // invert ssim_map_dst 1 - ssim_map_dst
    // cv::subtract(cv::Scalar::all(1.0), ssim_map_dst, ssim_map_dst);

    // set zero if value in ssim_map_dst too small
    // ssim_map_dst.setTo(0, ssim_map_dst < 0.9);

    // ssim_map_dst normalize to 0, 255
    cv::normalize(ssim_map_dst, ssim_map_dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);

    // convert to grayscale
    cv::cvtColor(ssim_map_dst, ssim_map_dst, cv::COLOR_BGR2GRAY);

    // convert to CV_8UC1
    // ssim_map_dst.convertTo(ssim_map_dst, CV_8UC1);

    // convert to compatible type for otsu
//    cv::cvtColor(ssim_map_dst, ssim_map_dst, cv::COLOR_BGR2GRAY);
//
//    // otsu threshold
//    cv::threshold(ssim_map_dst, ssim_map_dst, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
//
////    // erosion & dilation
//    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
//    cv::erode(ssim_map_dst, ssim_map_dst, element);
//    cv::dilate(ssim_map_dst, ssim_map_dst, element);

    return mssim;
}

void detectDifference(cv::Mat &img1, cv::Mat &img2, cv::Mat &img3,
                      int num_tiles_x, int num_tiles_y, std::vector<int> &tile_result) {
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);

    // convert to grayscale
    cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY);

    // convert to CV_8UC1
    diff.convertTo(diff, CV_8UC1);

    // otsu threshold
    cv::threshold(diff, diff, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // erosion & dilation
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::erode(diff, diff, element);
    cv::dilate(diff, diff, element);

    double scale_x = 1.0;
    double scale_y = 1.0;

    cv::Mat low_res;
    cv::resize(diff, low_res, cv::Size(), scale_x, scale_y, cv::INTER_NEAREST);


    double tile_w = low_res.cols/(double)num_tiles_x;
    double tile_h = low_res.rows/(double)num_tiles_y;

    // store tiles as int, each tile is represented by a single value
    std::vector<int> tiles;

    for (int y = 0; y < num_tiles_y; y++) {
        for (int x = 0; x < num_tiles_x; x++) {
            cv::Mat tile = low_res(cv::Rect(x * tile_w, y * tile_h, tile_w, tile_h));
            cv::Scalar mean = cv::mean(tile);

            // if tile diff > 0.0 then fill location in low_res with white
            if (mean[0] > 0.0) {
                // mark tile as white
                tiles.push_back(1);
            } else {
                // mark tile as black
                tiles.push_back(0);
            }
        }
    }

//    // detect holes and hole radius
//    std::vector<cv::Point> holes;
//    std::vector<int> hole_radius;
//
//    for (int y = 0; y < num_tiles_y; y++) {
//        for (int x = 0; x < num_tiles_x; x++) {
//            if (tiles[y * num_tiles_x + x] == 0) {
//                int radius = 0;
//                bool found = false;
//                for (int r = 1; r < 3; r++) {
//                    int count = 0;
//                    for (int i = -r; i <= r; i++) {
//                        for (int j = -r; j <= r; j++) {
//                            if (x + i >= 0 && x + i < num_tiles_x && y + j >= 0 && y + j < num_tiles_y) {
//                                if (tiles[(y + j) * num_tiles_x + (x + i)] == 1) {
//                                    count++;
//                                }
//                            }
//                        }
//                    }
//                    if (count > r * 7) {
//                        radius = r;
//                        found = true;
//                        break;
//                    }
//                }
//                if (found) {
//                    holes.push_back(cv::Point(x, y));
//                    hole_radius.push_back(radius);
//                }
//            }
//        }
//    }

//    // fill holes with white tiles
//    for (int i = 0; i < holes.size(); i++) {
//        cv::Point p = holes[i];
//        int r = hole_radius[i];
//        for (int i = -r; i <= r; i++) {
//            for (int j = -r; j <= r; j++) {
//                if (p.x + i >= 0 && p.x + i < num_tiles_x && p.y + j >= 0 && p.y + j < num_tiles_y) {
//                    tiles[(p.y + j) * num_tiles_x + (p.x + i)] = 1;
//                }
//            }
//        }
//    }

    // draw tiles on low_res
    for (int y = 0; y < num_tiles_y; y++) {
        for (int x = 0; x < num_tiles_x; x++) {
            if (tiles[y * num_tiles_x + x] == 1) {
                cv::rectangle(low_res, cv::Rect(x * tile_w, y * tile_h, tile_w+1, tile_h+1), cv::Scalar(255, 255, 255), -1);
            } else {
                cv::rectangle(low_res, cv::Rect(x * tile_w, y * tile_h, tile_w+1, tile_h+1), cv::Scalar(0, 0, 0), -1);
            }
        }
    }

    tile_result = tiles;

    // back to original size (be exact)
    cv::resize(low_res, diff, cv::Size(img1.cols, img1.rows), 0, 0, cv::INTER_NEAREST);

    // log diff_size and img1_size
    std::cout << "diff size: " << diff.size() << std::endl;
    std::cout << "img1 size: " << img1.size() << std::endl;

    // copy low res to img3
    diff.copyTo(img3);
}

std::tuple<std::vector<std::string>, std::vector<std::string>> read_images_to_stack(
        const std::string &input_folder_path, std::vector<cv::Mat> &images) {

    // log reading images
    LOG_INFO("Reading images to stack from folder: " + input_folder_path);

    std::vector<std::string> image_paths;
    std::vector<std::string> image_names;

    for(auto& entry : std::filesystem::directory_iterator(input_folder_path)) {
        if (entry.is_regular_file()) {

            auto image_name = entry.path().filename().string();
            auto image_path = entry.path().string();

            if (image_path.find(".jpg") != std::string::npos ||
                image_path.find(".png") != std::string::npos ||
                image_path.find(".jpeg") != std::string::npos) {

                image_names.push_back(image_name);
                image_paths.push_back(image_path);
//                if(paths.has_value()) {
//                    paths.value().push_back(image_path);
//                }
            }
        }
    }

    // read images in parallel
    std::mutex m;
    std::for_each(
#ifdef __APPLE__
            // std::execution::par, TODO find out how to run parallel for
#else
            std::execution::par,
#endif
            std::begin(image_paths),
            std::end(image_paths), [&](auto image_path) {
                // log file name
                LOG_INFO("reading image: " + image_path);

                // read the image and convert it to floating point to reduce rounding errors
                cv::Mat img = cv::imread(image_path);
                {
                    std::lock_guard lock(m);
                    images.push_back(img);
                }
            });

    return {image_paths, image_names};
}

void align_image(const cv::Mat &ref_img, const cv::Mat &img, const cv::Mat &img_orig, cv::Mat &img_aligned) {

    int warp_mode = cv::MOTION_HOMOGRAPHY;//cv::MOTION_EUCLIDEAN;

    // set a 2x3 or 3x3 transformation matrix depending on the motion model.
    cv::Mat transformation_matrix;

    // the transformation matrix (initialized with the identity matrix)
    if(warp_mode == cv::MOTION_HOMOGRAPHY) {
        transformation_matrix = cv::Mat::eye(3, 3, CV_32F /*32bit float*/);
    } else {
        transformation_matrix = cv::Mat::eye(2, 3, CV_32F /*32bit float*/);
    }

    // number of iterations for ECC
    int number_of_iterations = 100;

    // specify the threshold of the increment in the correlation coefficient
    // between two iterations
    double termination_eps = 1e-8;

    // define termination criteria
    cv::TermCriteria criteria (cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                               number_of_iterations, termination_eps);

    // finally, we apply the ecc algorithm. the transformation result is stored in transformation_matrix.
    double correlation_value = cv::findTransformECC(
            ref_img,
            img,
            transformation_matrix,
            warp_mode,
            criteria
    );

    if (correlation_value == -1) {
        LOG_ERROR("The execution of the ECC algorithm was interrupted. The correlation value is going to be minimized."
                  " Check the warp initialization and/or the size of images.");
    } else {
        LOG_INFO(" -> transformation found. the correlation value is: " + std::to_string(correlation_value));
    }

    img_aligned = cv::Mat(img_orig.rows, img_orig.cols, img_orig.type());

    float scale =  (float)img_orig.cols / (float)img.cols;

    if(warp_mode == cv::MOTION_HOMOGRAPHY) {

        // apply image scale difference to homography matrix
        // first, define the scale transform (diagonal matrix, inverse of performed image scaling)
        cv::Mat scale_transform = cv::Mat::eye(3, 3, CV_32F /*float*/);
        scale_transform.at<float>(0, 0) = scale;
        scale_transform.at<float>(1, 1) = scale;
        scale_transform.at<float>(2, 2) =  1.0f;

        // then apply the scale transform to the homography matrix (from the left)
        // and the inverse of the scale transform (from the right)
        // -> the matrix from the left is for transforming the homography matrix to match the image scaling
        // -> the matrix from the right is for making sure the homography matrix is scaled back to the original size
        //    since the output image would be shown in the original (small) size
        //
        // the formula: H_scaled =  S * H * S_inv
        //
        transformation_matrix = scale_transform * transformation_matrix * scale_transform.inv();

    } else {
        // rescale the translation part matrix to match the size of the original image
        cv::Mat translation_part = transformation_matrix.col(2);
        translation_part.at<float>(0) *= scale;
        translation_part.at<float>(1) *= scale;
    }

    // use the transformation matrix to warp the source image to the destination image
    if(warp_mode == cv::MOTION_HOMOGRAPHY) {
        cv::warpPerspective(img_orig, img_aligned, transformation_matrix, img_orig.size(),
                            cv::INTER_LINEAR + cv::WARP_INVERSE_MAP
        );
    } else {
        warpAffine(img_orig, img_aligned, transformation_matrix, img_orig.size(),
                   cv::INTER_LINEAR + cv::WARP_INVERSE_MAP
        );
    }
}

void align_images(const std::vector<cv::Mat> &images, std::vector<cv::Mat> &aligned_images, const float resize_factor_for_alignment) {

    std::vector<cv::Mat> images_grayscale(images.size());

    std::vector<cv::Mat> images_float = images;
    std::atomic<size_t> i = 0;

// read images in parallel
    std::mutex m;
    std::for_each(
#ifdef __APPLE__
            // std::execution::par, TODO find out how to run parallel for
#else
            std::execution::par,
#endif
            std::begin(images),
            std::end(images), [&](auto img) {

                size_t local_i = i++;
                if (img.type() != CV_32FC3) {
                    LOG_INFO("images to stack must be of type CV_32FC3! converting image[{}] from {} to CV_32FC3...",
                             local_i, type2str(img.type())
                    );
                    cv::Mat img_32fc3;
                    img.convertTo(img_32fc3, CV_32FC3);
                    {
                        std::lock_guard<std::mutex> lock(m);
                        images_float[local_i] = img_32fc3;
                    }
                }

// convert to grayscale and rescale
                {
                    cv::Mat img_resized;
                    cv::resize(img, img_resized, cv::Size(), resize_factor_for_alignment, resize_factor_for_alignment);
                    cv::Mat img_gray;
                    cvtColor(img_resized, img_gray, cv::COLOR_BGR2GRAY);
                    {
                        std::lock_guard<std::mutex> lock(m);
                        images_grayscale[local_i] = img_gray;
                    }
                }
            });

// use first image as reference image
    cv::Mat ref_img = images_grayscale[0];

// image to analyze
    cv::Mat img = images_grayscale[1];

// align images
    i = 0;
    std::for_each(
#ifdef __APPLE__
            // std::execution::par, TODO find out how to run parallel for
#else
            std::execution::par,
#endif
            std::begin(images),
            std::end(images), [&](auto img) {
                size_t local_i = i++;
                LOG_INFO("aligning image: {}", local_i);
                cv::Mat img_aligned;
                align_image(ref_img, images_grayscale[local_i], images_float[local_i], img_aligned);
                {
                    std::lock_guard<std::mutex> lock(m);
                    aligned_images.push_back(img_aligned);
                }
            });
}

std::tuple<cv::Mat,std::vector<cv::Mat>> stack_images(const std::vector<cv::Mat> &images, const float grayscale_factor) {

    std::vector<cv::Mat> images_grayscale(images.size());

    std::vector<cv::Mat> images_float = images;
    std::atomic<size_t> i = 0;

    // read images in parallel
    std::mutex m;
    std::for_each(
#ifdef __APPLE__
            // std::execution::par, TODO find out how to run parallel for
#else
            std::execution::par,
#endif
            std::begin(images),
            std::end(images), [&](auto img) {

                size_t local_i = i++;
                if (img.type() != CV_32FC3) {
                    LOG_INFO("images to stack must be of type CV_32FC3! converting image[{}] from {} to CV_32FC3...",
                             local_i, type2str(img.type())
                    );
                    cv::Mat img_32fc3;
                    img.convertTo(img_32fc3, CV_32FC3);
                    {
                        std::lock_guard<std::mutex> lock(m);
                        images_float[local_i] = img_32fc3;
                    }
                }

                // convert to grayscale and rescale
                {
                    cv::Mat img_resized;
                    cv::resize(img, img_resized, cv::Size(), grayscale_factor, grayscale_factor);
                    cv::Mat img_gray;
                    cvtColor(img_resized, img_gray, cv::COLOR_BGR2GRAY);
                    {
                        std::lock_guard<std::mutex> lock(m);
                        images_grayscale[local_i] = img_gray;
                    }
                }
            });

    // use first image as reference image
    cv::Mat ref_img = images_grayscale[0];

    // image to analyze
    cv::Mat img = images_grayscale[1];

    // list of aligned image
    std::vector<cv::Mat> images_aligned;

    // align images
    i = 0;
    std::for_each(
#ifdef __APPLE__
            // std::execution::par, TODO find out how to run parallel for
#else
            std::execution::par,
#endif
            std::begin(images),
            std::end(images), [&](auto img) {
                size_t local_i = i++;
                LOG_INFO("aligning image: {}", local_i);
                cv::Mat img_aligned;
                align_image(ref_img, images_grayscale[local_i], images_float[local_i], img_aligned);
                {
                    std::lock_guard<std::mutex> lock(m);
                    images_aligned.push_back(img_aligned);
                }
            });

//    // replace humans in aligned images via hog detector
//    i = 0;
//    std::for_each(std::execution::par, std::begin(images_aligned),
//                  std::end(images_aligned), [&](auto img) {
//        size_t local_i = i++;
//        LOG_INFO("detecting humans in image: {}", local_i);
//        std::vector<cv::Rect> humans;
//
//        // shrink image to speed up detection
//        float shrinking_scale = 0.125;
//        cv::Mat img_small;
//        cv::resize(img, img_small, cv::Size(), shrinking_scale, shrinking_scale);
//
//        detect_humans(img_small, humans);
//
//        // scale humans rects
//        std::for_each(std::execution::par, std::begin(humans),
//                      std::end(humans), [&](auto &rect) {
//            rect.x /= shrinking_scale;
//            rect.y /= shrinking_scale;
//            rect.width /= shrinking_scale;
//            rect.height /= shrinking_scale;
//        });
//
//        if(humans.size() > 0) {
//            LOG_INFO("found {} humans in image: {}", humans.size(), local_i);
//            for(auto human : humans) {
//                cv::rectangle(img, human, cv::Scalar(0, 0, 255), -1);
//                cv::imwrite("human_" + std::to_string(local_i) + ".jpg", img);
//            }
//        }
//    });

    // stack images
    cv::Mat stacked_image = cv::Mat::zeros(
            images_aligned[0].rows, images_aligned[0].cols,
            images_aligned[0].type());

    // add images to stacked image with weight 1.0/n
    size_t n = images_aligned.size();

//    // init range for parallel loop 0..n-1
//    std::vector<int> v(n);
//    std::iota(v.begin(), v.end(), 0);

    for(size_t idx = 0; idx < n; idx++) {
        // log image name
        LOG_INFO("stacking image: {}", idx);
        // add weighted image to stacked image
        addWeighted(stacked_image, 1.0, images_aligned[idx], 1.0/n, 0, stacked_image);
    }

    return {stacked_image, images_aligned};
}

void compute_diff(cv::Mat img_1, cv::Mat img_2, cv::Mat& diff_img, double img_scale, int kernel_size, std::vector<int>& tiles, int num_tiles_x, int num_tiles_y) {

    // img_1, img_2 to images
    std::vector<cv::Mat> images;
    images.push_back(img_1);
    images.push_back(img_2);

    // scale images, ensure that images have the same size
    cv::Size img_size = images[0].size();
    for (auto &img : images) {
        if (img.size() != img_size) {
            // log
            LOG_INFO("Resizing image to {}x{}", img_size.width, img_size.height);
            // scale image to the size of the first image
            cv::resize(img, img, img_size);
        }
    }

    // apply scale value
    if(img_scale != 1.0) {
        for (auto &img : images) {
            cv::resize(img, img, cv::Size(), img_scale, img_scale);
        }
    }

    // check that kernel size is within limits of 0 and 13
    if(kernel_size < 0 || kernel_size > 13) {
        LOG_ERROR("kernel-size must be in range [0, 13]");
        CLOSE_APP(APP_ERROR | APP_ARGS_ERROR);
    }

    // blur images
    if(kernel_size > 0) {
        // check if kernel size is odd
        if(kernel_size % 2 == 0) {
            kernel_size++;
            // log
            LOG_INFO("Kernel size not odd, increasing it to {}", kernel_size);
        }

        for (auto &img : images) {
            cv::GaussianBlur(img, img,
                             cv::Size(kernel_size, kernel_size),
                             0, 0);
        }
    }

    cv::Mat diff_map;

    auto s = computeMSSIM(images[0], images[1], diff_map);

//    // write ssim map to file
//    cv::imwrite("ssim-map.png", diff_map);

    detectDifference(images[0], images[1], diff_map, num_tiles_x, num_tiles_y, tiles);

    // scale diff_map to original image size
    cv::resize(diff_map, diff_map, img_size);

//    // write diff_map transparent on top of image 1 to file
//    cv::Mat diff_map_transparent = images[1].clone();
//    cv::cvtColor(diff_map, diff_map, cv::COLOR_GRAY2BGR);
//    cv::addWeighted(diff_map, 0.5, diff_map_transparent, 0.5, 0.0, diff_map_transparent);
//    cv::imwrite("diff-map-overlay.png", diff_map_transparent);

    // print results
    LOG_INFO("SSIM: rgb=({},{},{})", s(0), s(1), s(2));

    diff_img = diff_map;
}

cv::Mat calculate_sharpness(const cv::Mat& image, int max_levels, int kernel_size, double scale, double delta) {
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
        cv::Laplacian( level, laplacian, CV_64F, kernel_size, /*scale*/scale, /*delta*/delta);
        cv::Mat energy;
        cv::convertScaleAbs(laplacian, energy);
        // Upscale to the original size
        cv::resize(energy, energy,
                   cv::Size(image.cols, image.rows),
                   0, 0, cv::INTER_LINEAR);
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