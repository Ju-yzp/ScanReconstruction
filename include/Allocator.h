#ifndef ALLOCATOR_H_
#define ALLOCATOR_H_

#include <immintrin.h>
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <Constants.h>
#include <GlobalSettings.h>
#include <Octree.h>
#include <Types.h>
#include <Utils.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <fstream>

namespace ScanReconstruction {

class ReconstructionPipeline {
public:
    ReconstructionPipeline(std::shared_ptr<GlobalSettings> global_settings);

    void fusion(const Points& points, const Eigen::Matrix4f& camera_pose, bool allocate = true);

    void raycast(Points& points, const Eigen::Matrix4f& camera_pose);

    void exportToSTL(const std::string& filename) { tree_->exportToSTL(filename, voxel_size_); }

private:
    void allocateMemoryForVoxels(
        const Points& points, const Eigen::Matrix4f& camera_pose, bool allocate = true);

    void integrate(const Points& points, const Eigen::Matrix4f& camera_pose);

    void voxelDownSample(
        const Points& origin_points, Points& processed_points, Eigen::Matrix4f camera_pose);

    float readFromSDFInterpolated(
        const Eigen::Vector3f& point, const Eigen::Vector3i& blockPos,
        const Voxel* current_voxel_block);

    inline float readFromSDFUninterpolated(
        const Eigen::Vector3f& point, const Eigen::Vector3i& blockPos,
        const Voxel* current_voxel_block) {
        if (!current_voxel_block) return 1.0f;
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

    // Allocator allcator_;

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
    Octree* tree_;
};
}  // namespace ScanReconstruction

#endif
