#ifndef SETTINGS_H_
#define SETTINGS_H_

#include <opencv2/opencv.hpp>

namespace SufaceRestruction {
struct Settings {
    // 彩色图像尺寸
    cv::Size2i rgb_imageSize{0, 0};

    // 深度图像尺寸
    cv::Size2i depth_imageSize{0, 0};

    // 体素分辨率
    float voxelSize{0.005f};

    // 截断距离
    float mu{0.02f};

    // 判断是否需要重新生成点云的阈值
    float regenerate_pointcloud_threahold{0.1f};

    // 朝向权重
    float orientation_weight{1.0f};

    // 平移权重
    float translation_weight{0.6f};
};
}  // namespace SufaceRestruction
#endif  // SETTINGS_H_
