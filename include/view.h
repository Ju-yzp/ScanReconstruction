#ifndef VIEW_H_
#define VIEW_H_

#include <opencv2/opencv.hpp>

#include <calibrationParams.h>

namespace surface_restruction {
struct View {
    // 深度图像
    cv::Mat depth;

    // 彩色图像
    cv::Mat rgb;

    // 上一帧的彩色图像
    cv::Mat prve_rgb;

    // 相机标定参数
    RGBDCalibrationParams calibrationParams;
};
}  // namespace surface_restruction

#endif  // VIEW_H_
