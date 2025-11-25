#ifndef TRACKING_STATE_H_
#define TRACKING_STATE_H_

#include <Eigen/Eigen>

struct TrackingState {
    Eigen::Matrix4f pose_d;
};

#endif
