#include <imuKinematicObserver.h>
#include <synchronizer.h>

#include <libobsensor/h/ObTypes.h>
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Context.hpp>
#include <libobsensor/hpp/Device.hpp>
#include <libobsensor/hpp/Pipeline.hpp>

#include <chrono>
#include <functional>
#include <memory>
#include <thread>

int main() {
    Synchronizer synchronizer;

    ob::Context context;
    std::shared_ptr<ob::DeviceList> device_list = context.queryDeviceList();

    if (device_list->deviceCount() < 1) {
        std::cout << "Please check device is correctly connect width compute" << std::endl;
        return 0;
    }

    std::shared_ptr<ob::Device> device = device_list->getDevice(0);

    std::cout << "Default using the first device" << std::endl;

    std::shared_ptr<ob::Config> config = std::make_shared<ob::Config>();
    std::shared_ptr<ob::Pipeline> pipe = std::make_shared<ob::Pipeline>(device);

    config->enableGyroStream(OB_GYRO_FS_1000dps, OB_SAMPLE_RATE_500_HZ);
    config->enableAccelStream(OB_ACCEL_FS_2g, OB_SAMPLE_RATE_500_HZ);

    Eigen::Vector3d gravity(0, 0, -9.8);
    Eigen::Vector3d init_bg(00.000224886, -7.61038e-05, -0.000742259);
    Eigen::Vector3d init_ba(-0.165205, 0.0926887, 0.0058049);
    ImuKinematicObserver iko(gravity, init_ba, init_bg);

    synchronizer.set_and_start_thread(
        std::bind(&ImuKinematicObserver::computeNewPose, &iko, std::placeholders::_1));

    pipe->start(
        config,
        std::bind(&Synchronizer::accpectNewSensorData, &synchronizer, std::placeholders::_1));

    std::shared_ptr<ob::CameraParamList> cp = device->getCalibrationCameraParamList();
    while (true) std::this_thread::sleep_for(std::chrono::milliseconds(10));
}
