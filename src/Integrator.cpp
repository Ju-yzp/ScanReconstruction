#include <Integrator.h>
#include <cmath>

// tbb
#include <VoxelHash.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <Eigen/Core>
#include <cstddef>

namespace ScanReconstruction {
Integrator::Integrator(std::shared_ptr<GlobalSettings> global_settings)
    : mu_(global_settings->mu),
      max_weight_(global_settings->max_weight),
      voxel_size_(global_settings->voxel_size),
      height_(global_settings->height),
      width_(global_settings->width),
      k_(global_settings->k) {
    for (int x = 0; x < EXPAND_BOUND; ++x)
        for (int y = 0; y < EXPAND_BOUND; ++y)
            for (int z = 0; z < EXPAND_BOUND; ++z)
                coord_offsets_.emplace_back(Eigen::Vector3i(x, y, z));
}

void Integrator::integrateDepthIntoScene(
    const Points& points, const Eigen::Matrix4f& camera_pose,
    const std::vector<HashEntry>& updated_entries, std::shared_ptr<Scene> scene) {
    const Eigen::Matrix3f r = camera_pose.block(0, 0, 3, 3).transpose();
    const Eigen::Vector3f t = camera_pose.block(0, 3, 3, 1);

    Timer timer("integrate");

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, updated_entries.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t id = range.begin(); id != range.end(); ++id) {
                const HashEntry& current_hashEntry = updated_entries[id];
                Voxel* current_voxel_block = scene->get_voxel_block(current_hashEntry.ptr);
                const Eigen::Vector3i& blockPos = current_hashEntry.pos;
                for (size_t i = 0; i < coord_offsets_.size(); ++i) {
                    const Eigen::Vector3i& offset = coord_offsets_[i];
                    int localId = offset.x() + offset.y() * EXPAND_BOUND +
                                  offset.z() * EXPAND_BOUND * EXPAND_BOUND;
                    Eigen::Vector3f voxel_in_world = blockPos.cast<float>() * VOXEL_BLOCK_SIZE;
                    voxel_in_world += (offset + Eigen::Vector3i(-1, -1, -1)).cast<float>();
                    voxel_in_world *= voxel_size_;
                    Eigen::Vector3f voxel_in_camera = r * (voxel_in_world - t);
                    Voxel& current_voxel = current_voxel_block[localId];

                    Eigen::Vector3f pointImage = k_ * voxel_in_camera;
                    pointImage /= pointImage(2);

                    if (pointImage(0) < 0 || pointImage(0) > float(width_ - 1) ||
                        pointImage(1) < 0 || pointImage(1) > float(height_ - 1))
                        continue;

                    const Eigen::Vector3f& point = points[size_t(
                        (int)(pointImage(0) + 0.5) + (int)(pointImage(1) + 0.5) * width_)];
                    if (std::isnan(point(0))) continue;
                    float eta = point(2) - voxel_in_camera(2);

                    // 不在截断区域内,跳过不更新
                    if (std::abs(eta) < mu_) continue;

                    // 更新sdf值
                    float old_sdf = current_voxel.get_sdf();
                    float old_weight = current_voxel.get_weight();
                    // float old_sdf = current_voxel.sdf;
                    // float old_weight = current_voxel.weight;
                    if (old_weight == max_weight_) continue;
                    float new_sdf = std::min(1.0f, eta / mu_);
                    float new_weight = 1.0f;

                    new_sdf = old_weight * old_sdf + new_weight * new_sdf;
                    new_weight = new_weight + old_weight;
                    new_sdf /= new_weight;

                    new_weight = std::min(new_weight, max_weight_);

                    current_voxel.set_sdf(new_sdf);
                    current_voxel.set_weight(new_weight);

                    // current_voxel.sdf = new_sdf;
                    // current_voxel.weight = new_weight;
                }
            }
        });
}
}  // namespace ScanReconstruction
