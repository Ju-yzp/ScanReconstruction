#include <coreEngine.h>
#include <opencv2/core/hal/interface.h>
#include <libobsensor/hpp/Frame.hpp>
#include <memory>
#include <opencv2/core/types.hpp>

namespace surface_reconstruction {
void CoreEngine::statrtCamera() {
    try {
        ob::Context cxt;
        std::shared_ptr<ob::DeviceList> dev_list = cxt.queryDeviceList();
        if (dev_list->deviceCount() == 0) {
            return;
        }

        dev_ = dev_list->getDevice(0);

        pipe_ = std::make_shared<ob::Pipeline>(dev_);

        std::shared_ptr<ob::Config> config = std::make_shared<ob::Config>();

        config->enableVideoStream(OB_STREAM_COLOR, 1920, 1080, 30, OB_FORMAT_BGRA);

        config->enableVideoStream(OB_STREAM_DEPTH, 640, 576, 30, OB_FORMAT_Y16);

        pipe_->start(config, [&](std::shared_ptr<ob::FrameSet> frames) {
            if (frames->colorFrame() && frames->depthFrame()) {
                DepthAdnRGB images;
                int depth_rows = frames->depthFrame()->height();
                int depth_cols = frames->depthFrame()->width();
                int rgb_rows = frames->colorFrame()->height();
                int rgb_cols = frames->colorFrame()->width();

                images.depth.create(cv::Size2i(depth_cols, depth_rows), CV_16U);
                images.rgb.create(cv::Size2i(rgb_cols, rgb_rows), CV_8UC4);

                images.depth.data = (uchar*)frames->depthFrame()->data();
                images.rgb.data = (uchar*)frames->colorFrame()->data();
            }
        });

    } catch (ob::Error& e) {
        std::cerr << "function:" << e.getName() << "\nargs:" << e.getArgs()
                  << "\nmessage:" << e.getMessage() << "\ntype:" << e.getExceptionType()
                  << std::endl;
    }
}

void stratProcessThread() {}
}  // namespace surface_reconstruction
