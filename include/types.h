#ifndef TYPES_H_
#define TYPES_H_

#include <Eigen/Eigen>

struct IMU {
    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
    uint64_t timestamp;
};

struct Voxel {
    short sdf;
    short weight;
};

#endif
