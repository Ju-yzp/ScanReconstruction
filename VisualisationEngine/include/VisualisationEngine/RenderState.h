#ifndef RENDER_STATE_H_
#define RENDER_STATE_H_

#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>

struct RenderState {
    RenderState(cv::Size imgSize) {
        raycastResult = cv::Mat(imgSize.height, imgSize.width, CV_32FC3);
    }
    cv::Mat raycastResult;
};

#endif
