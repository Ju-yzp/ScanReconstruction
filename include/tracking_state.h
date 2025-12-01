#ifndef TRACKING_STATE_H_
#define TRACKING_STATE_H_

// eigen
#include <Eigen/Core>
#include <Eigen/Eigen>

// cpp
#include <Eigen/Geometry>
#include <memory>

namespace SufaceRestruction {
class TrackingState {
public:
    TrackingState(
        int height, int weight, float orientation_weight, float translation_weight,
        float regenerate_pointcloud_threahold)
        : height_(height),
          width_(weight),
          orientation_weight_(orientation_weight),
          translation_weight_(translation_weight),
          regenerate_pointcloud_threahold_(regenerate_pointcloud_threahold) {
        pointcloud_ = std::make_shared<Eigen::Vector4f>(new Eigen::Vector4f[height_ * width_]);
    }

    bool NeedRegenerateNewPointcloud() {
        Eigen::Matrix4f transform =
            generate_camera_in_localmap_.inverse() * current_camera_in_localmap_;

        Eigen::AngleAxisf angleAxis(transform.block<3, 3>(0, 0));
        float score = std::abs(angleAxis.angle()) * orientation_weight_ +
                      transform.block<3, 1>(0, 3).norm() * translation_weight_;
        return score > regenerate_pointcloud_threahold_;
    }

    std::shared_ptr<Eigen::Vector4f> get_pointcloud() { return pointcloud_; }

    int get_height() const { return height_; }

    int get_width() const { return width_; }

    Eigen::Matrix4f get_current_camera_in_localmap() const { return current_camera_in_localmap_; }

    void set_current_camera_in_localmap(const Eigen::Matrix4f pose) {
        current_camera_in_localmap_ = pose;
    }

    Eigen::Matrix4f get_generate_camera_in_localmap() { return generate_camera_in_localmap_; }

    void set_generate_camera_in_localmap(const Eigen::Matrix4f pose) {
        generate_camera_in_localmap_ = pose;
    }

private:
    // 点云数据以及对应的图像和高度信息
    std::shared_ptr<Eigen::Vector4f> pointcloud_;
    int height_;
    int width_;

    // 生成的点云所对应的相机在子地图中的位姿
    Eigen::Matrix4f generate_camera_in_localmap_{Eigen::Matrix4f::Identity()};

    // 相机在子地图中的位姿
    Eigen::Matrix4f current_camera_in_localmap_{Eigen::Matrix4f::Identity()};

    // 相机位姿中朝向变化对于是否需要重新进行光线投射生成新点云的线性权重影响
    float orientation_weight_;

    // 相机位姿中位置变化对于是否需要重新进行光线投射生成新点云的线性权重影响
    float translation_weight_;

    // 判断是否需要重新生成点云的阈值
    float regenerate_pointcloud_threahold_;
};
}  // namespace SufaceRestruction

#endif  // TRACKING_STATE_H_
