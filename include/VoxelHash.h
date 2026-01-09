#ifndef VOXEL_HASH_H_
#define VOXEL_HASH_H_

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ScanReconstruction {

constexpr int VOXEL_BLOCK_SIZE = 8;
constexpr int VOXEL_BLOCK_SIZE3 = 512;
constexpr uint32_t SDF_HASH_MASK = 0xfffff;
constexpr int EXPAND_BOUND = 10;
constexpr int VOLUME = 1000;

struct Voxel {
    // Voxel(float sdf_v, float weight_v) : sdf(sdf_v), weight(weight_v) {}
    Voxel(int16_t sdf_v, uint16_t weight_v) : sdf(sdf_v), weight(weight_v) {}
    Voxel() = default;

    inline float get_weight() const {
        constexpr float scale = 100.0f / (float)std::numeric_limits<uint16_t>::max();
        return static_cast<float>(weight) * scale;
    }

    inline float get_sdf() const {
        using Lim = std::numeric_limits<int16_t>;
        constexpr float scale = 1.0f / (static_cast<float>(Lim::max()) + 1.0f);
        return static_cast<float>(sdf) * scale;
    }

    inline void set_sdf(float new_sdf) {
        using Lim = std::numeric_limits<int16_t>;
        sdf = static_cast<int16_t>(std::clamp(
            new_sdf * static_cast<float>(Lim::max()), static_cast<float>(Lim::min()),
            static_cast<float>(Lim::max())));
    }

    inline void set_weight(float new_weight) {
        using Lim = std::numeric_limits<uint16_t>;
        weight = static_cast<uint16_t>(std::clamp(
            std::round(new_weight * static_cast<float>(Lim::max())), static_cast<float>(Lim::min()),
            static_cast<float>(Lim::max())));
    }

    inline int64_t get_sdfWithInt64() const { return static_cast<int64_t>(sdf); }

    inline int64_t get_weightWithInt64() const { return static_cast<int64_t>(weight); }

    int16_t sdf;
    uint16_t weight;

    // float sdf;
    // float weight;
};

struct HashEntry {
    // 体素块的体素块世界坐标
    Eigen::Vector3i pos = Eigen::Vector3i::Zero();
    // 发生冲突时，指向的下一个entry的偏移量
    int offset = -1;
    // 相对于体素块内存首地址的偏移量
    int ptr = -1;
};

struct Vector3iHash {
    std::size_t operator()(const Eigen::Vector3i& voxelBlockPos) const {
        return (((uint)voxelBlockPos(0) * 73856093u) ^ ((uint)voxelBlockPos(1) * 19349669u) ^
                ((uint)voxelBlockPos(2) * 83492791u)) &
               (uint)SDF_HASH_MASK;
    }
};

struct VoxelHash {
    VoxelHash(size_t num) {
        voxel_ptr = new Voxel[num * VOLUME];
        entries_ptr = new HashEntry[num];
    }

    ~VoxelHash() {
        delete[] voxel_ptr;
        delete[] entries_ptr;
    }

    // 哈希函数
    static inline int getHashIndex(Eigen::Vector3i voxelBlockPos) {
        return (((uint)voxelBlockPos(0) * 73856093u) ^ ((uint)voxelBlockPos(1) * 19349669u) ^
                ((uint)voxelBlockPos(2) * 83492791u)) &
               (uint)SDF_HASH_MASK;
    }

    static inline void initVoxelBlock(Voxel* voxel_block) {
        for (int i = 0; i < VOLUME; ++i) {
            Voxel& current_voxel = voxel_block[i];
            current_voxel.set_sdf(1.0f);
            current_voxel.set_weight(0.0f);
            // current_voxel.sdf = 1.0f;
            // current_voxel.weight = 0.0f;
        }
    }

    static inline Eigen::Vector3i posToBlockPos(Eigen::Vector3i& point) {
        return Eigen::Vector3i(point.x() >> 3, point.y() >> 3, point.z() >> 3);
    }

    static inline int get_block_offset(
        const Eigen::Vector3i blockPos, const Eigen::Vector3i point) {
        Eigen::Vector3i offset = point - blockPos * VOXEL_BLOCK_SIZE;
        return (offset.z() << 6) + (offset.y() << 3) + offset.x();
    }

    Voxel* voxel_ptr;
    HashEntry* entries_ptr;
};
}  // namespace ScanReconstruction

#endif
