#ifndef SCAN_RECONSTRUCTION_PIPELINE_H_
#define SCAN_RECONSTRUCTION_PIPELINE_H_

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <memory>
#include <vector>

#include <GlobalSettings.h>
#include <RenderedView.h>
#include <TrackingState.h>
#include <Types.h>
#include <scene.h>

namespace ScanReconstruction {
class Pipeline {
public:
    Pipeline(std::shared_ptr<GlobalSettings> global_settings);

    void fusion(TrackingState* ts, bool allocate = false);

    void raycast(RenderedView* rv, Scene* scene);

    void reset();

private:
    void allocateMemory(TrackingState* ts, bool allocate = false);

    void integrate(TrackingState* ts);

    void voxelDownSample(std::vector<Eigen::Vector3f>& filter_points, TrackingState* ts);

    std::vector<std::pair<Eigen::Vector3f, VoxelBlock*>> need_integrated_list_;

    std::vector<uint64_t> need_allocate_list_;

    float voxel_size_;

    float mu_;

    int height_, width_;

    float viewFrustum_max_, viewFrustum_min_;

    Eigen::Matrix3f k_;

    Image ray_map_;

    std::vector<Eigen::Vector3f> offsets_;
};
}  // namespace ScanReconstruction

#endif
