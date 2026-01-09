#ifndef SCENE_H_
#define SCENE_H_

#include <GlobalSettings.h>
#include <VoxelHash.h>
#include <memory>
#include <optional>

namespace ScanReconstruction {

class Scene {
public:
    Scene(std::shared_ptr<GlobalSettings> global_settings);

    std::optional<HashEntry> insert(Eigen::Vector3i& blockPos);

    Voxel* get_voxel_block(int id) { return voxel_hash_.voxel_ptr + VOLUME * id; }

    float readFromSDFInterpolated(const Eigen::Vector3f& point);

    float readWithConfidenceFromSDFInterpolated(const Eigen::Vector3f& point);

    inline float readFromSDFUninterpolated(const Eigen::Vector3f& point) {
        // return readVoxel(Eigen::Vector3i(point.array().floor().cast<int>())).get_sdf();
        return readVoxel(Eigen::Vector3i(point.array().floor().cast<int>())).get_sdf();
    }

    const Voxel readVoxel(Eigen::Vector3i point);

    const Voxel* readVoxelBlock(Eigen::Vector3i blockPos);

    int get_free_entries_num() { return free_entries_num_; }

    int get_entries_num() { return entries_num_; }

private:
    VoxelHash voxel_hash_;

    float voxel_size_;

    int entries_num_;

    int free_entries_num_;
};

};  // namespace ScanReconstruction
#endif
