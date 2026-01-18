#ifndef TYPES_H_
#define TYPES_H_

#include <Constants.h>
#include <Eigen/Eigen>
#include <chrono>
#include <iostream>

namespace ScanReconstruction {

struct alignas(4) Voxel {
    short sdf;
    unsigned short depth_weight;
};

struct alignas(64) VoxelBlock {
    short sdf[VOXEL_BLOCK_SIZE3];
    unsigned short weight[VOXEL_BLOCK_SIZE3];
};

const uint64_t MASK_X = 0x1249249249249249;
const uint64_t MASK_Y = 0x2492492492492492;
const uint64_t MASK_Z = 0x4924924924924924;

inline bool isValid(const Eigen::Vector3i& pos) {
    constexpr int32_t boundary = 1 << MAX_OCTREE_DEPTH;
    constexpr int32_t half = boundary >> 1;

    return (pos.array() >= -half && pos.array() < half).all();
}

inline uint64_t encode(const Eigen::Vector3i& pos) {
    constexpr int32_t boundary = 1 << MAX_OCTREE_DEPTH;
    constexpr int32_t half = boundary >> 1;

    uint64_t x = static_cast<uint64_t>(std::clamp(pos.x() + half, 0, boundary - 1));
    uint64_t y = static_cast<uint64_t>(std::clamp(pos.y() + half, 0, boundary - 1));
    uint64_t z = static_cast<uint64_t>(std::clamp(pos.z() + half, 0, boundary - 1));

    return _pdep_u64(x, MASK_X) | _pdep_u64(y, MASK_Y) | _pdep_u64(z, MASK_Z);
}

inline Eigen::Vector3i decode(uint64_t code) {
    constexpr int32_t half = 1 << (MAX_OCTREE_DEPTH - 1);

    uint64_t ux = _pext_u64(code, MASK_X);
    uint64_t uy = _pext_u64(code, MASK_Y);
    uint64_t uz = _pext_u64(code, MASK_Z);

    return Eigen::Vector3i(
        static_cast<int>(ux) - half, static_cast<int>(uy) - half, static_cast<int>(uz) - half);
}

struct Triangle {
    Eigen::Vector3f normal;  // 面片法向量：决定了模型在光照下的明暗
    Eigen::Vector3f v[3];    // 三个顶点坐标
    uint16_t attribute;  // 属性字节：在标准 STL 中通常为0，但在某些软件中可以用来存颜色

    // 默认构造，初始化法线为0
    Triangle() : normal(0, 0, 0), attribute(0) {}
};

inline float shortToFloat(short sdf) {
    using Lim = std::numeric_limits<short>;
    constexpr float scale = 1.0f / (static_cast<float>(Lim::max()) + 1.0f);
    return static_cast<float>(sdf) * scale;
}

inline float ushortToFloat(unsigned short depth_weight) {
    constexpr float scale = 100.0f / (float)std::numeric_limits<uint16_t>::max();
    return static_cast<float>(depth_weight) * scale;
}

inline short floatToShort(float new_sdf) {
    using Lim = std::numeric_limits<int16_t>;
    return static_cast<short>(std::clamp(
        new_sdf * static_cast<float>(Lim::max()), static_cast<float>(Lim::min()),
        static_cast<float>(Lim::max())));
}

inline unsigned short floatToUshort(float new_depth_weight) {
    using Lim = std::numeric_limits<unsigned short>;
    return static_cast<unsigned short>(std::clamp(
        std::round(new_depth_weight * static_cast<float>(Lim::max())),
        static_cast<float>(Lim::min()), static_cast<float>(Lim::max())));
}

inline float readSDFByVoxelBlock(const Voxel* voxel_block, Eigen::Vector3i position) {
    position += Eigen::Vector3i(1, 1, 1);
    return shortToFloat(voxel_block
                            [position.x() + position.y() * EXPANDED_VOXEL_BLOCK_SIZE +
                             position.z() * EXPANDED_VOXEL_BLOCK_SIZE * EXPANDED_VOXEL_BLOCK_SIZE]
                                .sdf);
}

inline Voxel readVoxelByVoxelBlock(const Voxel* voxel_block, Eigen::Vector3i position) {
    position += Eigen::Vector3i(1, 1, 1);
    return voxel_block
        [position.x() + position.y() * EXPANDED_VOXEL_BLOCK_SIZE +
         position.z() * EXPANDED_VOXEL_BLOCK_SIZE * EXPANDED_VOXEL_BLOCK_SIZE];
}

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
