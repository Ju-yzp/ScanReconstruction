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
    uint8_t red;
    uint8_t green;
    uint8_t bule;
    uint8_t color_weight;
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
        current_voxel.red = current_voxel.green = current_voxel.bule = 0;
        current_voxel.color_weight = 0;
    }
};

inline bool isValid(Eigen::Vector3i pos, int depth_limit) {
    uint32_t boundary = 1U << depth_limit;

    return (static_cast<uint32_t>(pos.x()) < boundary) &&
           (static_cast<uint32_t>(pos.y()) < boundary) &&
           (static_cast<uint32_t>(pos.z()) < boundary);
}

inline uint64_t encode(const Eigen::Vector3i& pos) {
    uint64_t x = static_cast<uint64_t>(pos.x());
    uint64_t y = static_cast<uint64_t>(pos.y());
    uint64_t z = static_cast<uint64_t>(pos.z());

    return _pdep_u64(x, 0x1249249249249249) | _pdep_u64(y, 0x2492492492492492) |
           _pdep_u64(z, 0x4924924924924924);
}

inline Eigen::Vector3i decode(uint64_t code) {
    return Eigen::Vector3i(
        static_cast<int>(_pext_u64(code, 0x1249249249249249)),
        static_cast<int>(_pext_u64(code, 0x2492492492492492)),
        static_cast<int>(_pext_u64(code, 0x4924924924924924)));
}

class Allocator {
public:
    Allocator(size_t reversed_size, std::function<void(void*)>);

    bool allocate(uint64_t code);

    Voxel* accessVoxelBlock(uint64_t code);

    bool hasAllocate(uint64_t code) const;

private:
    void* address_;

    size_t reversed_size_;

    std::function<void(void*)> initialize_callback_;

    size_t page_size_;

    std::unique_ptr<std::atomic<uint64_t>[]> bit_map_;
};

class ReconstructionPipeline {
public:
    ReconstructionPipeline(std::shared_ptr<GlobalSettings> global_settings, int depth_limit);

    void fusion(const Points& points, const Eigen::Matrix4f& camera_pose);

    void raycast(Points& points, const Eigen::Matrix4f& camera_pose);

private:
    void allocateMemoryForVoxels(const Points& points, const Eigen::Matrix4f& camera_pose);

    void integrate(const Points& points, const Eigen::Matrix4f& camera_pose);

    void voxelDownSample(const Points& origin_points, Points& processed_points);

    float readFromSDFInterpolated(const Eigen::Vector3f& point);

    inline float readFromSDFUninterpolated(const Eigen::Vector3f& point) {
        return shortToFloat(readVoxel(Eigen::Vector3i(point.array().floor().cast<int>())).sdf);
    }

    const Voxel readVoxel(Eigen::Vector3i point);

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
