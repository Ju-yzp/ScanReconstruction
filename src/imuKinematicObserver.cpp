#include <imuKinematicObserver.h>
#include <cmath>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

ImuKinematicObserver::ImuKinematicObserver(
    const Eigen::Vector3d bg, const Eigen::Vector3d ba, const Eigen::Vector3d gravity)
    : bg_(bg), ba_(ba), gravity_(gravity) {
    p_ = v_ = Eigen::Vector3d::Zero();
    timestamp_ = 0.0;
}

Sophus::SE3d ImuKinematicObserver::computeNewPose(const IMU& imu) {
    constexpr double scale = 1.0 / 1000000;
    double dt = (double)(imu.timestamp - timestamp_) * scale;

    p_ = p_ + v_ * dt + 0.5 * gravity_ * dt * dt + 0.5 * (r_ * (imu.accel - ba_)) * dt * dt;
    v_ = v_ + r_ * (imu.accel - ba_) * dt + gravity_ * dt;
    r_ = r_ * Sophus::SO3d::exp((imu.gyro - bg_) * dt);
    timestamp_ = imu.timestamp;

    return Sophus::SE3d(r_, p_);
}
