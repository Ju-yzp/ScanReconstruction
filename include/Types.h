#ifndef TYPES_H_
#define TYPES_H_

#include <Eigen/Eigen>
#include <chrono>
#include <iostream>

namespace ScanReconstruction {

struct Timer {
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
    Timer(std::string n) : name(n), start(std::chrono::high_resolution_clock::now()) {}
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << name << " took " << dur.count() << " ms" << std::endl;
    }
};

using Normals = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;
using Points = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;

struct IMU {
    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
    uint64_t timestamp;
};

enum class TrackingResult { LOST = -1, POOR = 0, GOOD = 1 };

enum class MapType {
    PRIMARY_LOCAL_MAP = 0,
    NEW_LOCAL_MAP = 1,
    LOOP_CLOSURE = 2,
    RELOCALISATION = 3,
    LOST = 4,
    LOST_NEW = 5
};
}  // namespace ScanReconstruction
#endif
