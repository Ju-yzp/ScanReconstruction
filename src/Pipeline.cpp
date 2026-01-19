#include <Pipeline.h>
#include <hp_utils.h>

#include <Eigen/Core>

#include <unordered_set>
#include <vector>

namespace ScanReconstruction {
constexpr uint32_t SDF_HASH_MASK = 0xfffff;

struct Vector3iHash {
    std::size_t operator()(const Eigen::Vector3i& voxelBlockPos) const {
        return (((uint)voxelBlockPos(0) * 73856093u) ^ ((uint)voxelBlockPos(1) * 19349669u) ^
                ((uint)voxelBlockPos(2) * 83492791u)) &
               (uint)SDF_HASH_MASK;
    }
};

Pipeline::Pipeline(std::shared_ptr<GlobalSettings> global_settings)
    : voxel_size_(global_settings->voxel_size),
      mu_(global_settings->mu),
      height_(global_settings->height),
      width_(global_settings->width),
      k_(global_settings->k) {
    ray_map_.resize((size_t)(height_ * width_));
    Eigen::Matrix3f inv_k = k_.inverse();
    for (int y = 0; y < height_; ++y)
        for (int x = 0; x < width_; ++x)
            ray_map_[(size_t)(y * width_ + x)] = inv_k * Eigen::Vector3f(float(x), float(y), 1.0f);
}

void Pipeline::reset() {
    need_integrated_list_.clear();
    need_allocate_list_.clear();
}

void Pipeline::allocateMemory(TrackingState* ts, bool allocate) {
    const Eigen::Matrix3f r = ts->current_pose.block(0, 0, 3, 3);
    const Eigen::Vector3f t = ts->current_pose.block(0, 3, 3, 1);

    std::unordered_set<uint64_t> filter_allocated;
    const float oneOverVoxelSize = 1.0f / (voxel_size_ * VOXEL_BLOCK_SIZE);
    std::vector<Eigen::Vector3f> filter_points;
    voxelDownSample(filter_points, ts);

    for (auto& point : filter_points) {
        int nstep = 0;
        float norm = 0.0f;

        Eigen::Vector3f point_in_camera = point, direction;
        norm = point_in_camera.norm();

        Eigen::Vector3f point_s =
            (r * point_in_camera * (1.0f - mu_ / norm) + t) * oneOverVoxelSize;
        Eigen::Vector3f point_e =
            (r * point_in_camera * (1.0f + mu_ / norm) + t) * oneOverVoxelSize;

        direction = point_e - point_s;
        nstep = (int)ceil(2.0f * direction.norm());
        direction /= (float)(nstep - 1);

        for (int i = 0; i < nstep; ++i) {
            Eigen::Vector3i blockPos(
                (int)std::floor(point_s(0)), (int)std::floor(point_s(1)),
                (int)std::floor(point_s(2)));
            point_s += direction;
        }
    }
}

void Pipeline::integrate(TrackingState* ts) {
    const Eigen::Matrix3f inv_r = ts->current_pose.block(0, 0, 3, 3).transpose();
    const Eigen::Vector3f t = ts->current_pose.block(0, 3, 3, 1);

    // TODO:約束在P核上進行更新sdf和weight任務，執行計算密集型任務
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<size_t>(0, need_integrated_list_.size()),
        [&](const oneapi::tbb::blocked_range<size_t>& range) {
            std::vector<float> depth_pixel, depth_voxel;
            depth_pixel.resize(VOXEL_BLOCK_SIZE3);
            depth_voxel.resize(VOXEL_BLOCK_SIZE3);
            for (size_t id = range.begin(); id != range.end(); ++id) {
                auto [position, voxel_block] = need_integrated_list_[id];
                if (!voxel_block) continue;
                for (int i = 0; i < (int)VOXEL_BLOCK_SIZE3; ++i) {
                    Eigen::Vector3f voxel_in_camera = inv_r * (position + offsets_[i]) + t;
                    Eigen::Vector3f reproject_point = k_ * voxel_in_camera;
                    reproject_point /= reproject_point(2);
                    float depth;
                    depth_pixel[i] = depth;
                    depth_voxel[i] = voxel_in_camera(2);
                }

                integrateWithVoxelBlock(voxel_block, depth_voxel, depth_pixel, mu_);
            }
        });
}

void Pipeline::voxelDownSample(std::vector<Eigen::Vector3f>& filter_points, TrackingState* ts) {
    const Eigen::Matrix3f r = ts->current_pose.block(0, 0, 3, 3);
    const Eigen::Vector3f t = ts->current_pose.block(0, 3, 3, 1);
    const float oneOverVoxelSize = 1.0f / (voxel_size_ * VOXEL_BLOCK_SIZE * 0.9f);
    std::unordered_set<Eigen::Vector3i, Vector3iHash> voxel_map;
    for (const auto& point : ts->current_points) {
        if (std::isnan(point(0))) continue;
        Eigen::Vector3f tmp = r * point + t;
        const Eigen::Vector3i coord = (tmp.array() * oneOverVoxelSize).floor().cast<int>();
        if (voxel_map.insert(coord).second) filter_points.push_back(point);
    }
}
}  // namespace ScanReconstruction
