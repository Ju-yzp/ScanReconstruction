#ifndef PIXEL_UTILS_HPP_
#define PIXEL_UTILS_HPP_

#include <opencv2/core/hal/interface.h>
#include <cassert>
#include <opencv2/opencv.hpp>

inline void depthFilterSubsample(const cv::Mat* input, cv::Mat* output) {
    assert(input->type() == CV_32F);
    int rows_in = input->rows;
    int cols_in = input->cols;
    int rows_out = rows_in / 2;
    int cols_out = cols_in / 2;

    output->resize(rows_out, rows_in);
    const float* data_in = (float*)input->data;
    float* data_out = (float*)output->data;

    for (int y{0}; y < output->rows; ++y) {
        for (int x{0}; x < output->cols; ++x) {
            int x_src = x * 2;
            int y_src = y * 2;
            int num = 0;
            float sum = 0.0f;
            float v;

            v = data_in[y_src * cols_in + x_src];
            if (v > 0.0f) {
                num++;
                sum += v;
            }

            v = data_in[y_src * cols_in + x_src + 1];
            if (v > 0.0f) {
                num++;
                sum += v;
            }

            v = data_in[(y_src + 1) * cols_in + x_src];
            if (v > 0.0f) {
                num++;
                sum += v;
            }

            v = data_in[(y_src + 1) * cols_in + x_src + 1];
            if (v > 0.0f) {
                num++;
                sum += v;
            }

            if (num > 0)
                v = sum / (float)num;
            else
                v = 0.0f;

            data_out[y * cols_out + x] = v;
        }
    }
}
#endif
