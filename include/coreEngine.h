#ifndef CORE_ENGINE_H_
#define CORE_ENGINE_H_

#include <atomic>
#include <memory>
#include <queue>
#include <thread>

// orbbec sdk
#include <libobsensor/h/ObTypes.h>
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Context.hpp>
#include <libobsensor/hpp/Device.hpp>
#include <libobsensor/hpp/Frame.hpp>
#include <libobsensor/hpp/Pipeline.hpp>
#include <libobsensor/hpp/Sensor.hpp>

// opencv
#include <opencv2/opencv.hpp>

namespace surface_reconstruction {
// TODO:为了快速开发验证，直接将相机写入引擎，不考虑弱耦合
class CoreEngine {
public:
    // 启动处理图像线程
    void stratProcessThread();

    // 启动相机线程
    void statrtCamera();

private:
    struct SpinLock {
        std::atomic_flag flag = ATOMIC_FLAG_INIT;
        void lock() {
            while (flag.test_and_set(std::memory_order_acquire)) {
            }
        }

        void unlock() { flag.clear(std::memory_order_release); }
    };

    struct DepthAdnRGB {
        cv::Mat depth;
        cv::Mat rgb;
    };

    // 图像处理线程
    std::thread processThread_;

    // 设备
    std::shared_ptr<ob::Device> dev_;

    // 管线
    std::shared_ptr<ob::Pipeline> pipe_;

    // 消费者队列
    std::queue<DepthAdnRGB> producerQueue_;

    // 生产者队列
    std::queue<DepthAdnRGB> resumerQueue_;

    // 保护消费者队列的访问
    SpinLock spin_lock_;
};
}  // namespace surface_reconstruction

#endif
