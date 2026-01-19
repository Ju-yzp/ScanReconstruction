#ifndef SCAN_RECONSTRUCTION_SCENE_H_
#define SCAN_RECONSTRUCTION_SCENE_H_

#include <Types.h>
#include <atomic>
#include <cstdint>

namespace ScanReconstruction {
struct Node {
    std::atomic<uint64_t> space_status_map;
};

class Scene {
public:
    VoxelBlock* findVoxelBlock(Eigen::Vector3i position) const;

    VoxelBlock* findVoxelBlock(uint64_t code) const;

    void add_nodes(const std::vector<uint64_t>& morton_codes);

    static uint64_t encode(const Eigen::Vector3i& point_in_voxel_world);

    static Eigen::Vector3f decode(uint64_t morton_code, float voxel_size);

private:
};
}  // namespace ScanReconstruction

#endif
