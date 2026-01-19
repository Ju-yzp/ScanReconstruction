#ifndef SCAN_RECONSTRUCTION_TRACKING_STATE_H_
#define SCAN_RECONSTRUCTION_TRACKING_STATE_H_

#include <Types.h>
#include <Eigen/Core>

namespace ScanReconstruction {
struct TrackingState {
    Image current_points;
    Eigen::Matrix4f current_pose;
};
}  // namespace ScanReconstruction

#endif
