#include <PixelUtils.h>
#include <Raycaster.h>
#include <Eigen/Core>

// cpp
#include <VoxelHash.h>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

// tbb
#include <tbb/blocked_range2d.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>

namespace ScanReconstruction {
Raycaster::Raycaster(std::shared_ptr<GlobalSettings> global_settings)
    : viewFrustum_max_(global_settings->viewFrustum_max),
      viewFrustum_min_(global_settings->viewFrustum_min),
      width_(global_settings->width),
      height_(global_settings->height),
      k_(global_settings->k),
      voxel_size_(global_settings->voxel_size),
      mu_(global_settings->mu) {
    if (mu_ / voxel_size_ < 2.0f) throw std::runtime_error("mu must larger than voxel size");
    const Eigen::Matrix3f inv_k = k_.inverse();

    ray_map_.resize(size_t(width_ * height_), Eigen::Vector3f::Zero());
    for (int y = 0; y < height_; ++y)
        for (int x = 0; x < width_; ++x)
            ray_map_[(size_t)(y * width_ + x)] = inv_k * Eigen::Vector3f(float(x), float(y), 1.0f);
}

void Raycaster::allocateVoxelblocks(
    const Points& points, const Eigen::Matrix4f& camera_pose, std::shared_ptr<Scene> scene) {
    const Eigen::Matrix3f r = camera_pose.block(0, 0, 3, 3);
    const Eigen::Vector3f t = camera_pose.block(0, 3, 3, 1);

    const float oneOverVoxelSize = 1.0f / (voxel_size_ * VOXEL_BLOCK_SIZE);

    std::unordered_set<Eigen::Vector3i, Vector3iHash> need_updated_list;

    Timer timer("allocate memory");

    Points filtered_points;
    voxelFilter(points, filtered_points);
    for (size_t id = 0; id < filtered_points.size(); ++id) {
        int nstep = 0;
        float norm = 0.0f;

        Eigen::Vector3f point_in_camera = filtered_points[id], direction;
        if (std::isnan(point_in_camera(0))) continue;
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
            if (need_updated_list.find(blockPos) == need_updated_list.end()) {
                auto result = scene->insert(blockPos);
                need_updated_list.insert(blockPos);
                if (result.has_value()) updated_hashEntries_.emplace_back(result.value());
            }
            point_s += direction;
        }
    }
}

void Raycaster::raycast(
    Points& points, const Eigen::Matrix4f& camera_pose, std::shared_ptr<Scene> scene) {
    const Eigen::Matrix3f r = camera_pose.block(0, 0, 3, 3);
    // const Eigen::Matrix3f inv_r = camera_pose.block(0, 0, 3, 3).inverse();
    const Eigen::Vector3f t = camera_pose.block(0, 3, 3, 1);

    const float oneOverVoxelSize = 1.0f / voxel_size_;
    const float step_scale = mu_ * oneOverVoxelSize;
    const float sdf_value_max = mu_ / voxel_size_ / 2.0f, sdf_value_min = -sdf_value_max;

    Timer timer("raycast");
    tbb::parallel_for(
        tbb::blocked_range2d<int>(0, height_, 10, 0, width_, 10),
        [&](const tbb::blocked_range2d<int>& range) {
            for (int y = range.rows().begin(); y < range.rows().end(); ++y) {
                Eigen::Vector3f* points_ptr = points.data() + y * width_;
                Eigen::Vector3f* ray_map_ptr = ray_map_.data() + y * width_;
                for (int x = range.cols().begin(); x < range.cols().end(); ++x) {
                    Eigen::Vector3f rayDirection = Eigen::Vector3f::Zero();

                    float totalLenght = 0.0f, totalLenghtMax = 0.0f, stepLen = 0.0f;
                    const Eigen::Vector3f& temp = ray_map_ptr[x];
                    Eigen::Vector3f pointE = temp * viewFrustum_max_;

                    Eigen::Vector3f point_e = (r * pointE + t) * oneOverVoxelSize;

                    Eigen::Vector3f pointS = temp * viewFrustum_min_;
                    Eigen::Vector3f point_s = (r * pointS + t) * oneOverVoxelSize;

                    totalLenght = pointS.norm() * oneOverVoxelSize;
                    totalLenghtMax = pointE.norm() * oneOverVoxelSize;

                    rayDirection = point_e - point_s;
                    rayDirection.normalize();

                    Eigen::Vector3f pt_result = point_s;

                    float sdf_v = 0.0f;

                    bool pointFound{false};

                    while (totalLenght < totalLenghtMax) {
                        sdf_v = scene->readFromSDFUninterpolated(pt_result);
                        if (sdf_v < sdf_value_max && sdf_v > sdf_value_min)
                            sdf_v = scene->readFromSDFInterpolated(pt_result);
                        if (sdf_v < 0.0f) break;
                        stepLen = std::max(sdf_v * step_scale, 1.0f);

                        pt_result += stepLen * rayDirection;
                        totalLenght += stepLen;
                    }

                    if (sdf_v < 0.0f) {
                        stepLen = sdf_v * step_scale;
                        pt_result += stepLen * rayDirection;

                        sdf_v = scene->readWithConfidenceFromSDFInterpolated(pt_result);

                        stepLen = sdf_v * step_scale;
                        pt_result += stepLen * rayDirection;
                        pointFound = true;
                    }

                    if (pointFound)
                        points_ptr[x] = pt_result * voxel_size_;
                    else
                        points_ptr[x](0) = std::numeric_limits<float>::quiet_NaN();
                }
            }
        });
}

void Raycaster::voxelFilter(const Points& input, Points& output) {
    std::unordered_set<Eigen::Vector3i, Vector3iHash> voxel_map;
    voxel_map.reserve(std::min(input.size(), (size_t)100000));

    output.clear();
    output.reserve(std::min(input.size(), (size_t)50000));

    const float oneOverVoxelSize = 1.0f / (voxel_size_ * VOXEL_BLOCK_SIZE);

    for (const auto& p : input) {
        if (std::isnan(p(0))) continue;
        const Eigen::Vector3i coord = (p.array() * oneOverVoxelSize).floor().cast<int>();
        if (voxel_map.insert(coord).second) {
            output.push_back(p);
        }
    }
}
}  // namespace ScanReconstruction
