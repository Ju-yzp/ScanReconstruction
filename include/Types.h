#ifndef TYPES_H_
#define TYPES_H_

#include <Eigen/Eigen>

namespace ScanReconstruction {

using Normals = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;
using Points = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;

struct IMU {
    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
    uint64_t timestamp;
};

struct Voxel {
    short sdf;
    short weight;
};

enum class TrackingResult { LOST = -1, POOR = 0, GOOD = 1 };
}  // namespace ScanReconstruction
#endif
