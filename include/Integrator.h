#ifndef INTEGRATOR_H_
#define INTEGRATOR_H_

#include <Scene.h>
#include <Types.h>
#include <Eigen/Core>
#include <memory>

namespace ScanReconstruction {
class Integrator {
public:
    Integrator(std::shared_ptr<GlobalSettings> global_settings);

    void integrateDepthIntoScene(
        const Points& points, const Eigen::Matrix4f& camera_pose,
        const std::vector<HashEntry>& updated_entries, std::shared_ptr<Scene> scene);

private:
    float mu_;

    float max_weight_;

    float voxel_size_;

    int height_, width_;

    Eigen::Matrix3f k_;

    std::vector<Eigen::Vector3i> coord_offsets_;
};
}  // namespace ScanReconstruction

#endif
