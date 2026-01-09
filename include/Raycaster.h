#ifndef RAYCASTER_H_
#define RAYCASTER_H_

#include <GlobalSettings.h>
#include <Scene.h>
#include <Types.h>
#include <Eigen/Core>
#include <memory>

namespace ScanReconstruction {
class Raycaster {
public:
    Raycaster(std::shared_ptr<GlobalSettings> global_settings);

    void allocateVoxelblocks(
        const Points& points, const Eigen::Matrix4f& camera_pose, std::shared_ptr<Scene> scene);

    void raycast(Points& points, const Eigen::Matrix4f& camera_pose, std::shared_ptr<Scene> scene);

    const std::vector<HashEntry>& get_updated_hashEntries() const { return updated_hashEntries_; }

    void reset_updated_hashEntries() { updated_hashEntries_.clear(); }

private:
    void voxelFilter(const Points& input, Points& output);

    float viewFrustum_max_, viewFrustum_min_;

    int width_, height_;

    Eigen::Matrix3f k_;
    float voxel_size_;
    float mu_;

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> ray_map_;

    std::vector<HashEntry> updated_hashEntries_;
};
}  // namespace ScanReconstruction
#endif
