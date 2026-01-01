#include <libobsensor/h/ObTypes.h>

#include <Eigen/Core>

#include <cstdint>
#include <libobsensor/hpp/Frame.hpp>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <synchronizer.h>

// TODO: 没有处理时间戳跳变的异常情况
void Synchronizer::accpectNewSensorData(std::shared_ptr<ob::FrameSet> frameSet) {
    std::unique_lock<std::mutex> global_lock(global_mutex_);

    for (int i = 0; i < frameSet->frameCount(); ++i) {
        auto frame = frameSet->getFrame(i);
        OBFrameType frame_type = frame->type();
        switch (frame_type) {
            case OB_FRAME_COLOR:
                color_data_.emplace_back(frame->as<ob::ColorFrame>());
                break;
            case OB_FRAME_DEPTH:
                depth_data_.emplace_back(frame->as<ob::DepthFrame>());
                break;
            case OB_FRAME_GYRO:
                gyro_data_.emplace_back(frame->as<ob::GyroFrame>());
                std::cout << (double)frame->systemTimeStampUs() << std::endl;
                ImuDataSync();
                break;
            case OB_FRAME_ACCEL:
                accel_data_.emplace_back(frame->as<ob::AccelFrame>());
                break;
            default:
                throw std::runtime_error("Unsupport sensor data type !");
        }
    }
}

Eigen::Vector3d Synchronizer::linearInterpolation(
    std::shared_ptr<ob::AccelFrame> start_frame, std::shared_ptr<ob::AccelFrame> end_frame,
    uint64_t timestamp) {
    OBAccelValue start = start_frame->value();
    OBAccelValue end = end_frame->value();
    Eigen::Vector3d start_accel{start.x, start.y, start.z}, end_accel{end.x, end.y, end.z};
    return start_accel +
           (end_accel - start_accel) /
               (double(end_frame->systemTimeStampUs()) - double(start_frame->systemTimeStampUs())) *
               (double(timestamp) - double(start_frame->systemTimeStampUs()));
}

void Synchronizer::ImuDataSync() {
    if (gyro_data_.size() < 8 || accel_data_.size() < 8) return;
    std::shared_ptr<ob::GyroFrame> gyro = gyro_data_[gyro_data_.size() - 1];
    std::shared_ptr<ob::AccelFrame> accel = accel_data_[accel_data_.size() - 1];

    uint64_t gyro_timestamp = gyro->systemTimeStampUs();
    uint64_t accel_timestamp = accel->systemTimeStampUs();
    if (gyro_timestamp >= accel_timestamp) return;

    int start_id = -1;
    if (gyro_data_[0]->systemTimeStampUs() < accel_data_[0]->systemTimeStampUs()) {
        uint64_t ts = accel_data_[0]->systemTimeStampUs();
        for (int i = 0; i < gyro_data_.size(); ++i) {
            std::shared_ptr<ob::GyroFrame> current_gyro = gyro_data_[i];
            uint64_t timestamp = current_gyro->systemTimeStampUs();
            if (timestamp >= ts) {
                start_id = i;
                break;
            }
        }

        if (start_id == -1) return;
    }

    start_id = 0;
    int j = start_id, i = 0;
    for (; i < accel_data_.size() - 1; ++i) {
        std::shared_ptr<ob::AccelFrame> next_accel = accel_data_[i + 1],
                                        current_accel = accel_data_[i];

        uint64_t next_timestamp = next_accel->systemTimeStampUs();
        uint64_t current_timestamp = current_accel->systemTimeStampUs();
        for (; j < gyro_data_.size(); ++j) {
            std::shared_ptr<ob::GyroFrame> current_gyro = gyro_data_[j];
            uint64_t timestamp = current_gyro->systemTimeStampUs();

            if (timestamp > next_timestamp) break;

            Eigen::Vector3d accel = linearInterpolation(current_accel, next_accel, timestamp);
            IMU imu;
            imu.accel = accel;
            imu.timestamp = timestamp;
            OBGyroValue gyro = current_gyro->value();
            imu.gyro(0) = gyro.x;
            imu.gyro(1) = gyro.y;
            imu.gyro(2) = gyro.z;
            imu_data_.emplace_back(imu);
        }
    }
    if (!imu_data_.empty()) {
        std::unique_lock<std::mutex> imu_lock(imu_mutex_);
        pending_imu_data_.insert(
            pending_imu_data_.end(), std::make_move_iterator(imu_data_.begin()),
            std::make_move_iterator(imu_data_.end()));
        imu_data_.clear();
        cv_.notify_one();
    }

    gyro_data_.erase(gyro_data_.begin(), gyro_data_.begin() + j);

    accel_data_.erase(accel_data_.begin(), accel_data_.begin() + i);
}
