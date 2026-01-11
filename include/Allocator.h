#ifndef ALLOCATOR_H_
#define ALLOCATOR_H_

#include <immintrin.h>
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include <GlobalSettings.h>
#include <Types.h>

namespace ScanReconstruction {

constexpr int VOXEL_BLOCK_SIZE = 8;
constexpr int VOXEL_BLOCK_SIZE3 = 512;
constexpr int EXPANDED_VOXEL_BLOCK_SIZE = 10;
constexpr int EXPANDED_VOXEL_BLOCK_SIZE3 = 1000;

struct alignas(4) Voxel {
    short sdf;
    unsigned short depth_weight;
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

static std::function<void(void*)> func = [](void* ptr) {
    Voxel* current_voxel_block = static_cast<Voxel*>(ptr);
    for (size_t i = 0; i < EXPANDED_VOXEL_BLOCK_SIZE3; ++i) {
        Voxel& current_voxel = current_voxel_block[i];
        current_voxel.sdf = std::numeric_limits<short>::max();
        current_voxel.depth_weight = std::numeric_limits<unsigned short>::min();
    }
};

const uint64_t MASK_X = 0x1249249249249249;
const uint64_t MASK_Y = 0x2492492492492492;
const uint64_t MASK_Z = 0x4924924924924924;

inline bool isValid(const Eigen::Vector3i& pos, int depth_limit) {
    int32_t boundary = 1 << depth_limit;
    int32_t half = boundary >> 1;

    auto check = [&](int v) { return v >= -half && v < half; };
    return check(pos.x()) && check(pos.y()) && check(pos.z());
}

inline uint64_t encode(const Eigen::Vector3i& pos, int depth_limit) {
    uint64_t offset = 1ULL << (depth_limit - 1);

    uint64_t x = static_cast<uint64_t>(pos.x() + offset);
    uint64_t y = static_cast<uint64_t>(pos.y() + offset);
    uint64_t z = static_cast<uint64_t>(pos.z() + offset);

    return _pdep_u64(x, MASK_X) | _pdep_u64(y, MASK_Y) | _pdep_u64(z, MASK_Z);
}

inline Eigen::Vector3i decode(uint64_t code, int depth_limit) {
    uint64_t offset = 1ULL << (depth_limit - 1);

    uint64_t ux = _pext_u64(code, MASK_X);
    uint64_t uy = _pext_u64(code, MASK_Y);
    uint64_t uz = _pext_u64(code, MASK_Z);

    return Eigen::Vector3i(
        static_cast<int>(ux - offset), static_cast<int>(uy - offset),
        static_cast<int>(uz - offset));
}

class Allocator {
public:
    Allocator(size_t reversed_size, std::function<void(void*)>);

    void allocate(std::vector<uint64_t>& codes);

    Voxel* accessVoxelBlock(const uint64_t& code);

    bool hasAllocate(const uint64_t& code) const;

    bool isVaild(const uint64_t& code) {
        if (code > static_cast<uint64_t>(reversed_size_)) return false;
        return true;
    }

private:
    void* address_;

    size_t reversed_size_;

    std::function<void(void*)> initialize_callback_;

    size_t page_size_;

    std::unique_ptr<std::atomic<uint64_t>[]> bit_map_;
};

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

class ReconstructionPipeline {
public:
    ReconstructionPipeline(std::shared_ptr<GlobalSettings> global_settings, int depth_limit);

    void fusion(const Points& points, const Eigen::Matrix4f& camera_pose);

    void raycast(Points& points, const Eigen::Matrix4f& camera_pose);

private:
    void allocateMemoryForVoxels(const Points& points, const Eigen::Matrix4f& camera_pose);

    void integrate(const Points& points, const Eigen::Matrix4f& camera_pose);

    void voxelDownSample(
        const Points& origin_points, Points& processed_points, Eigen::Matrix4f camera_pose);

    float readFromSDFInterpolated(
        const Eigen::Vector3f& point, const Eigen::Vector3i& blockPos,
        const Voxel* current_voxel_block);

    inline float readFromSDFUninterpolated(
        const Eigen::Vector3f& point, const Eigen::Vector3i& blockPos,
        const Voxel* current_voxel_block) {
        return shortToFloat(
            readVoxel(
                Eigen::Vector3i(point.array().floor().cast<int>()), blockPos, current_voxel_block)
                .sdf);
    }

    inline const Voxel readVoxel(
        Eigen::Vector3i point, const Eigen::Vector3i& blockPos, const Voxel* current_voxel_block) {
        Eigen::Vector3i offset = point - blockPos * VOXEL_BLOCK_SIZE;
        return readVoxelByVoxelBlock(current_voxel_block, offset);
    }

    Allocator allcator_;

    float mu_;

    float max_weight_;

    float voxel_size_;

    int height_, width_;

    Eigen::Matrix3f k_;

    float viewFrustum_max_, viewFrustum_min_;

    int depth_limit_;

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> ray_map_;

    std::vector<uint64_t> updated_list_;

    std::vector<Eigen::Vector3i> coord_offsets_;
};
}  // namespace ScanReconstruction

#endif
