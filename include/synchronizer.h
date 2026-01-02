#ifndef SYNCHRONIZER_H_
#define SYNCHRONIZER_H_

// orbbec
#include <Eigen/Core>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <libobsensor/hpp/Frame.hpp>

// cpp
#include <memory>
#include <mutex>
#include <sophus/se3.hpp>
#include <thread>
#include <vector>

#include <Types.h>

// TODO: 暂时是给Orbbec Femto Bolt写的，后期再修改下适配多设备软同步
class Synchronizer {
public:
    Synchronizer() {
        accel_data_.reserve(100);
        gyro_data_.reserve(100);
        depth_data_.reserve(20);
        color_data_.reserve(20);
        imu_data_.reserve(20);
        pending_imu_data_.reserve(20);
    }

    ~Synchronizer() {
        terminated_.store(true);
        cv_.notify_all();
        if (imu_process_thread_.joinable()) imu_process_thread_.join();
    }

    void accpectNewSensorData(std::shared_ptr<ob::FrameSet> frameSet);

    void set_and_start_thread(std::function<Sophus::SE3d(IMU& imu)> imu_callback) {
        imu_process_thread_ = std::thread([this, imu_callback]() {
            while (!terminated_.load()) {
                std::vector<IMU> local_imu_data;
                {
                    std::unique_lock<std::mutex> imu_lock(imu_mutex_);
                    cv_.wait(imu_lock, [this] {
                        return !pending_imu_data_.empty() || terminated_.load();
                    });
                    local_imu_data.swap(pending_imu_data_);
                }
                for (auto& imu : local_imu_data) {
                    std::cout << (double)imu.timestamp << std::endl;
                    imu_callback(imu);
                }
            }
        });
    }

private:
    void ImuDataSync();

    Eigen::Vector3d linearInterpolation(
        std::shared_ptr<ob::AccelFrame> start_frame, std::shared_ptr<ob::AccelFrame> end_frame,
        uint64_t timestamp);

    std::vector<std::shared_ptr<ob::AccelFrame>> accel_data_;

    std::vector<std::shared_ptr<ob::GyroFrame>> gyro_data_;

    std::vector<std::shared_ptr<ob::DepthFrame>> depth_data_;

    std::vector<std::shared_ptr<ob::ColorFrame>> color_data_;

    std::mutex global_mutex_;

    std::mutex imu_mutex_;

    std::vector<IMU> imu_data_;
    std::vector<IMU> pending_imu_data_;

    std::thread imu_process_thread_;

    std::condition_variable cv_;

    std::atomic<bool> terminated_{false};
};

#endif
