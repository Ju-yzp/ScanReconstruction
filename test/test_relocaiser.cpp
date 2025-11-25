#include <opencv2/core/hal/interface.h>
#include <../../FernRelocLib/include/FernRelocLib/Relocaliser.hpp>
#include <sophus/se3.hpp>

void convertUnsignedShortToFloat(cv::Mat& input, cv::Mat& output) {
    if (!input.empty()) {
        output = cv::Mat(input.rows, input.cols, CV_32F);
        for (int i{0}; i < input.rows; ++i) {
            for (int j{0}; j < input.rows; ++j) {
                output.at<float>(i, j) = (int)input.at<unsigned short>(i, j) / 1000.0;
            }
        }
    }
}

int main() {
    Eigen::Vector2i img_size;

    Eigen::Vector2f range;
    int type = CV_32F;
    cv::Mat img = cv::imread("../data/Frame1.png", cv::IMREAD_UNCHANGED);

    if (!img.empty()) {
        cv::Mat depth;
        convertUnsignedShortToFloat(img, depth);
        img_size(0) = img.rows;
        img_size(1) = img.cols;
        range(0) = 0.01f;
        range(1) = 5.0f;
        int NN[1];
        float distances[1];
        bool primaryTrackingSuccess = true;
        Sophus::SE3f pose;
        int primaryLocalMapIdx = -1;
        FernRelocLib::Relocaliser relocaliser(img_size, type, range, 0.1f, 1400, 4);
        relocaliser.processFrame(
            &depth, &pose, primaryLocalMapIdx, 1, NN, distances, primaryTrackingSuccess);

        img = cv::imread("../data/Frame2.png", cv::IMREAD_UNCHANGED);
        convertUnsignedShortToFloat(img, depth);

        primaryTrackingSuccess = true;
        bool hasAddedKeyframe = relocaliser.processFrame(
            &depth, &pose, primaryLocalMapIdx, 1, NN, distances, primaryTrackingSuccess);

        if (!hasAddedKeyframe) {
            std::cout << "This frame2 not added" << std::endl;
        }

        img = cv::imread("../data/Frame3.png", cv::IMREAD_UNCHANGED);
        convertUnsignedShortToFloat(img, depth);

        primaryTrackingSuccess = true;
        hasAddedKeyframe = relocaliser.processFrame(
            &depth, &pose, primaryLocalMapIdx, 1, NN, distances, primaryTrackingSuccess);

        if (!hasAddedKeyframe) {
            std::cout << "This frame3 not added" << std::endl;
        }
    }
}
