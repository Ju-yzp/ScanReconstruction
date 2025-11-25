#ifndef PIXEL_UTILS_H_
#define PIXEL_UTILS_H_

// opencv
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>

// eigen
#include <Eigen/Eigen>

// cpp
#include <cstdint>

inline void filterSubsample(cv::Mat* input, cv::Mat* output) {
    if (input->empty()) throw "Invoke filterSubsample() but image passed is empty";

    int type = input->type();
    output->create(input->rows / 2, input->cols / 2, input->type());

    for (int y{1}; y < output->rows - 1; ++y) {
        for (int x{1}; x < output->cols - 1; ++x) {
            if (type == CV_32F) {
                float pixel{0.0f};
                pixel += input->at<float>(y * 2, x * 2);
                pixel += input->at<float>(y * 2, x * 2 + 1);
                pixel += input->at<float>(y * 2 + 1, x * 2 + 1);
                pixel += input->at<float>(y * 2 + 1, x * 2);
                output->at<float>(y, x) = pixel / 4.0f;
            } else if (type == CV_32FC4) {
                cv::Vec4f pixel_in[4], pixel_out;
                pixel_in[0] = input->at<cv::Vec4f>(y * 2, x * 2);
                pixel_in[1] = input->at<cv::Vec4f>(y * 2 + 1, x * 2 + 1);
                pixel_in[2] = input->at<cv::Vec4f>(y * 2 + 1, x * 2);
                pixel_in[3] = input->at<cv::Vec4f>(y * 2, x * 2 + 1);

                pixel_out = (pixel_in[0] + pixel_in[1] + pixel_in[2] + pixel_in[3]) / 4.0f;
                output->at<cv::Vec4f>(y, x) = pixel_out;
            }
        }
    }
}

// 仅用于点云图、法向量图、深度图的下采样，因为彩色图不存在空洞(无效值或者不连续平面)，所以可以使用opencv接口
// 仅支持CV_32F、CV_32FC4格式
inline void filterSubsampleWithHoles(cv::Mat* input, cv::Mat* output) {
    if (input->empty())
        throw "Invoke filterSubsample() but image passed is empty or synatx of image is incorrect";

    int type = input->type();
    output->create(input->rows / 2, input->cols / 2, input->type());

    for (int y{0}; y < output->rows; ++y) {
        for (int x{0}; x < output->cols; ++x) {
            if (type == CV_32F) {
                float pixel_in[4], pixel_out{0.0f};
                pixel_in[0] = input->at<float>(y * 2, x * 2);
                pixel_in[1] = input->at<float>(y * 2 + 1, x * 2 + 1);
                pixel_in[2] = input->at<float>(y * 2 + 1, x * 2);
                pixel_in[3] = input->at<float>(y * 2, x * 2 + 1);

                int nVaildPoints{0};
                for (int k{0}; k < 4; ++k)
                    if (pixel_in[k] >= 0.0f) {
                        pixel_out += pixel_in[k];
                        ++nVaildPoints;
                    }

                if (nVaildPoints > 0) pixel_out /= (float)nVaildPoints;
                output->at<float>(y, x) = pixel_out;
            } else if (type == CV_32FC4) {
                cv::Vec4f pixel_in[4], pixel_out;
                pixel_in[0] = input->at<cv::Vec4f>(y * 2, x * 2);
                pixel_in[1] = input->at<cv::Vec4f>(y * 2 + 1, x * 2 + 1);
                pixel_in[2] = input->at<cv::Vec4f>(y * 2 + 1, x * 2);
                pixel_in[3] = input->at<cv::Vec4f>(y * 2, x * 2 + 1);

                int nVaildPoints{0};
                for (int k{0}; k < 4; ++k)
                    if (pixel_in[k](3) >= 0.0f) {
                        pixel_out += pixel_in[k];
                        ++nVaildPoints;
                    }

                if (nVaildPoints == 0) {
                    pixel_out(3) = -1.0f;
                    output->at<cv::Vec4f>(y, x) = pixel_out;
                    continue;
                }

                pixel_out /= (float)nVaildPoints;
                output->at<cv::Vec4f>(y, x) = pixel_out;
            }
        }
    }
}

// 将无符号16位的原始深度图转换为以m为单位的float图像
inline void convertShortToFloat(cv::Mat* input, cv::Mat* output, float scale) {
    output->create(input->rows, input->cols, CV_32F);

    for (int y{0}; y < output->rows; ++y) {
        for (int x{0}; x < output->cols; ++x) {
            int32_t pixel_in = input->at<uint16_t>(y, x);
            if (pixel_in > 0) output->at<float>(y, x) = (float)pixel_in / scale;
        }
    }
}

// 深度图、法向量图、点云图插值
inline Eigen::Vector4f interpolateBilinear_withHoles(cv::Mat img, Eigen::Vector2f coorinate) {
    Eigen::Vector2i imgPoint((int)floor(coorinate(0)), (int)floor(coorinate(1)));

    auto a = img.at<cv::Vec4f>(imgPoint(1), imgPoint(0));
    auto b = img.at<cv::Vec4f>(imgPoint(1), imgPoint(0) + 1);
    auto c = img.at<cv::Vec4f>(imgPoint(1) + 1, imgPoint(0));
    auto d = img.at<cv::Vec4f>(imgPoint(1) + 1, imgPoint(0) + 1);
    Eigen::Vector4f result;
    Eigen::Vector2f delta{coorinate(0) - imgPoint(0), coorinate(1) - imgPoint(1)};

    if (a(3) < 0.0f || a(3) < 0.0f || c(3) < 0.0f || d(3) < 0.0f) {
        result(0) = 0.0f;
        result(1) = 0.0f;
        result(2) = 0.0f;
        result(3) = -1.0f;
        return result;
    }

    result(0) = a(0) * (1.0f - delta(0)) * (1.0f - delta(1)) + b(0) * delta(0) * (1.0f - delta(1)) +
                c(0) * (1.0f - delta(0)) * delta(1) + d(0) * delta(0) * delta(1);

    result(1) = a(1) * (1.0f - delta(0)) * (1.0f - delta(1)) + b(1) * delta(0) * (1.0f - delta(1)) +
                c(1) * (1.0f - delta(0)) * delta(1) + d(1) * delta(0) * delta(1);

    result(2) = a(2) * (1.0f - delta(0)) * (1.0f - delta(1)) + b(2) * delta(0) * (1.0f - delta(1)) +
                c(2) * (1.0f - delta(0)) * delta(1) + d(2) * delta(0) * delta(1);

    result(3) = a(3) * (1.0f - delta(0)) * (1.0f - delta(1)) + b(3) * delta(0) * (1.0f - delta(1)) +
                c(3) * (1.0f - delta(0)) * delta(1) + d(3) * delta(0) * delta(1);
    return result;
}
#endif
