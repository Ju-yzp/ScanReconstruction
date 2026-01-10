#ifndef ALLCATOR_H_
#define ALLCATOR_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>

namespace ScanReconstruction {

constexpr int VOXEL_BLOCK_SIZE = 8;
constexpr int VOXEL_BLOCK_SIZE3 = 512;

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
    for (size_t i = 0; i < VOXEL_BLOCK_SIZE3; ++i) {
        Voxel& current_voxel = current_voxel_block[i];
        current_voxel.sdf = std::numeric_limits<short>::max();
        current_voxel.depth_weight = std::numeric_limits<unsigned short>::min();
        current_voxel.red = current_voxel.green = current_voxel.bule = 0;
        current_voxel.color_weight = 0;
    }
};

class Allcator {
public:
    Allcator(size_t reversed_size, std::function<void(void*)>);

    void allocate(uint32_t address);

    void deallocate(uint64_t address);

    Voxel* hasAllocated(uint32_t code);

private:
    void* address_;

    size_t reversed_size_;

    std::function<void(void*)> initialize_callback_;

    size_t page_size_;
};
}  // namespace ScanReconstruction

#endif
