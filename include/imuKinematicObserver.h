#ifndef IMU_KINEMATIC_OBSERVER_H_
#define IMU_KINEMATIC_OBSERVER_H_

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <types.h>

// 简单的IMU运动学观察器
class ImuKinematicObserver {
public:
    ImuKinematicObserver(
        const Eigen::Vector3d bg, const Eigen::Vector3d ba, const Eigen::Vector3d gravity);

    Sophus::SE3d computeNewPose(const IMU& imu);

private:
    // 零偏
    Eigen::Vector3d bg_;
    Eigen::Vector3d ba_;

    // 重力
    Eigen::Vector3d gravity_;

    // 累计量:旋转矩阵，位移量，速度
    Eigen::Vector3d v_;
    Eigen::Vector3d p_;
    Sophus::SO3d r_;

    // 上一刻的时间戳
    uint64_t timestamp_;
};
#endif
