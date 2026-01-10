#include <Allocator.h>

#include <Types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

// tbb
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <Eigen/Core>
#include <cstddef>

constexpr uint32_t SDF_HASH_MASK = 0xfffff;

struct Vector3iHash {
    std::size_t operator()(const Eigen::Vector3i& voxelBlockPos) const {
        return (((uint)voxelBlockPos(0) * 73856093u) ^ ((uint)voxelBlockPos(1) * 19349669u) ^
                ((uint)voxelBlockPos(2) * 83492791u)) &
               (uint)SDF_HASH_MASK;
    }
};

namespace ScanReconstruction {
Allocator::Allocator(size_t reversed_size, std::function<void(void*)> callback)
    : reversed_size_(reversed_size), initialize_callback_(callback) {
    page_size_ = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    // address_ =
    //     mmap(NULL, reversed_size_ * page_size_, PROT_READ | PROT_WRITE, MAP_ANONYMOUS, -1, 0);
    // if (address_ == MAP_FAILED)
    //     throw std::runtime_error("Failed to allcate virtual memory that user need ");
    page_size_ = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    size_t total_bytes = reversed_size_ * page_size_;

    // 調試打印，看看到底申請了多少
    std::cout << "Attempting to mmap: " << (total_bytes >> 30) << " GB of virtual memory."
              << std::endl;

    address_ = mmap(
        NULL, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1,
        0);

    if (address_ == MAP_FAILED) {
        perror("mmap failed");  // 這會打印出系統具體的錯誤原因 (如 Out of memory)
        throw std::runtime_error("Failed to allocate virtual memory.");
    }
    size_t bitmap_size = (reversed_size + 63) >> 6;
    bit_map_ = std::make_unique<std::atomic<uint64_t>[]>(bitmap_size);

    for (size_t i = 0; i < bitmap_size; ++i) bit_map_[i].store(0, std::memory_order_relaxed);
}

bool Allocator::allocate(uint64_t code) {
    if (code >= reversed_size_) return false;
    uint64_t offset = code >> 6;
    uint64_t bit_pos = code & 63;
    uint64_t mask = (1ULL << bit_pos);

    uint64_t old_val = bit_map_[offset].fetch_or(mask, std::memory_order_acq_rel);

    if (!(old_val & mask)) {
        initialize_callback_(static_cast<void*>((char*)address_ + code * page_size_));
        return true;
    }
    return false;
}

Voxel* Allocator::accessVoxelBlock(uint64_t code) {
    if (code > static_cast<uint64_t>(reversed_size_)) return nullptr;
    uint64_t offset = code >> 6;
    uint64_t bit_pos = code & 63;
    uint64_t mask = (1ULL << bit_pos);
    if (bit_map_[offset].load(std::memory_order_acquire) & mask)
        return reinterpret_cast<Voxel*>((char*)address_ + code * page_size_);
    else {
        // std::cout << "User access virtual memory that had no allcate pysiphy memory" <<
        // std::endl;
        return nullptr;
    }
}

bool Allocator::hasAllocate(uint64_t code) const {
    if (code >= reversed_size_) return false;
    uint64_t offset = code >> 6;
    uint64_t bit_pos = code & 63;
    uint64_t mask = (1ULL << bit_pos);

    return (bit_map_[offset].load(std::memory_order_acquire) & mask) != 0;
}

ReconstructionPipeline::ReconstructionPipeline(
    std::shared_ptr<GlobalSettings> global_settings, int depth_limit)
    : allcator_(static_cast<size_t>(pow(2, depth_limit * 3)), func),
      mu_(global_settings->mu),
      max_weight_(global_settings->max_weight),
      voxel_size_(global_settings->voxel_size),
      height_(global_settings->height),
      width_(global_settings->width),
      k_(global_settings->k),
      viewFrustum_max_(global_settings->viewFrustum_max),
      viewFrustum_min_(global_settings->viewFrustum_min),
      depth_limit_(depth_limit) {
    if (mu_ / voxel_size_ < 2.0f) throw std::runtime_error("mu must larger than voxel size");
    const Eigen::Matrix3f inv_k = k_.inverse();

    ray_map_.resize(size_t(width_ * height_), Eigen::Vector3f::Zero());
    for (int y = 0; y < height_; ++y)
        for (int x = 0; x < width_; ++x)
            ray_map_[(size_t)(y * width_ + x)] = inv_k * Eigen::Vector3f(float(x), float(y), 1.0f);

    for (int x = 0; x < EXPANDED_VOXEL_BLOCK_SIZE; ++x)
        for (int y = 0; y < EXPANDED_VOXEL_BLOCK_SIZE; ++y)
            for (int z = 0; z < EXPANDED_VOXEL_BLOCK_SIZE; ++z)
                coord_offsets_.emplace_back(Eigen::Vector3i(x, y, z));
}

void ReconstructionPipeline::fusion(const Points& points, const Eigen::Matrix4f& camera_pose) {
    allocateMemoryForVoxels(points, camera_pose);
    integrate(points, camera_pose);
    updated_list_.clear();
}

inline Eigen::Vector3i get_block_indices_stable(const Eigen::Vector3f& pos) {
    return Eigen::Vector3i(
        static_cast<int>(std::floor(pos.x() * 0.125f)),
        static_cast<int>(std::floor(pos.y() * 0.125f)),
        static_cast<int>(std::floor(pos.z() * 0.125f)));
}

inline float get_block_step_distance_stable(
    const Eigen::Vector3f& pos, const Eigen::Vector3f& invDir) {
    float bx = std::floor(pos.x() * 0.125f) * 8.0f;
    float by = std::floor(pos.y() * 0.125f) * 8.0f;
    float bz = std::floor(pos.z() * 0.125f) * 8.0f;

    float tx = ((invDir.x() > 0 ? bx + 8.0f : bx) - pos.x()) * invDir.x();
    float ty = ((invDir.y() > 0 ? by + 8.0f : by) - pos.y()) * invDir.y();
    float tz = ((invDir.z() > 0 ? bz + 8.0f : bz) - pos.z()) * invDir.z();

    if (tx <= 0) tx = 1e10f;
    if (ty <= 0) ty = 1e10f;
    if (tz <= 0) tz = 1e10f;

    return (std::min({tx, ty, tz}) * invDir).norm();
}

inline float readSDFByVoxelBlock(const Voxel* voxel_block, Eigen::Vector3i position) {
    position += Eigen::Vector3i(1, 1, 1);
    return shortToFloat(voxel_block
                            [position.x() + position.y() * EXPANDED_VOXEL_BLOCK_SIZE +
                             position.z() * EXPANDED_VOXEL_BLOCK_SIZE * EXPANDED_VOXEL_BLOCK_SIZE]
                                .sdf);
}

inline Voxel readVoxelByVoxelBlock(const Voxel* voxel_block, Eigen::Vector3i position) {
    position += Eigen::Vector3i(1, 1, 1);
    return voxel_block
        [position.x() + position.y() * EXPANDED_VOXEL_BLOCK_SIZE +
         position.z() * EXPANDED_VOXEL_BLOCK_SIZE * EXPANDED_VOXEL_BLOCK_SIZE];
}

void ReconstructionPipeline::raycast(Points& points, const Eigen::Matrix4f& camera_pose) {
    const Eigen::Matrix3f r = camera_pose.block(0, 0, 3, 3);
    const Eigen::Vector3f t = camera_pose.block(0, 3, 3, 1);

    const float oneOverVoxelSize = 1.0f / voxel_size_;
    const float step_scale = mu_ * oneOverVoxelSize;
    const float sdf_value_max = mu_ / voxel_size_ / 2.0f, sdf_value_min = -sdf_value_max;

    tbb::parallel_for(
        tbb::blocked_range2d<int>(0, height_, 10, 0, width_, 10),
        [&](const tbb::blocked_range2d<int>& range) {
            for (int y = range.rows().begin(); y < range.rows().end(); ++y) {
                Eigen::Vector3f* points_ptr = points.data() + y * width_;
                Eigen::Vector3f* ray_map_ptr = ray_map_.data() + y * width_;
                for (int x = range.cols().begin(); x < range.cols().end(); ++x) {
                    const Eigen::Vector3f& temp = ray_map_ptr[x];
                    Eigen::Vector3f pointE = temp * viewFrustum_max_;

                    Eigen::Vector3f point_e = (r * pointE + t) * oneOverVoxelSize;

                    Eigen::Vector3f pointS = temp * viewFrustum_min_;
                    Eigen::Vector3f point_s = (r * pointS + t) * oneOverVoxelSize;
                    float start_t = point_s.norm();
                    float totalLenght = pointS.norm() * oneOverVoxelSize;
                    float totalLenghtMax = pointE.norm() * oneOverVoxelSize;

                    Eigen::Vector3f rayDirection = (point_e - point_s).normalized();
                    Eigen::Vector3f invDir = rayDirection.cwiseInverse();

                    Eigen::Vector3f pt_result = point_s;
                    float sdf_v = 1.0f;
                    bool pointFound{false};

                    while (totalLenght < totalLenghtMax) {
                        Eigen::Vector3i blockPos = get_block_indices_stable(pt_result);
                        uint64_t code = encode(blockPos);

                        float t_to_boundary = get_block_step_distance_stable(pt_result, invDir);

                        if (!isValid(blockPos, depth_limit_) || !allcator_.hasAllocate(code)) {
                            float skip = t_to_boundary + 0.001f;
                            totalLenght += skip;

                            pt_result += skip * rayDirection;
                            continue;
                        }

                        sdf_v = readFromSDFUninterpolated(pt_result);
                        if (sdf_v < sdf_value_max && sdf_v > sdf_value_min) {
                            sdf_v = readFromSDFInterpolated(pt_result);
                        }

                        if (sdf_v < 0.0f) {
                            pointFound = true;
                            break;
                        }

                        float sdf_step = std::max(sdf_v * step_scale, 1.0f);
                        float safe_step = std::min(sdf_step, t_to_boundary + 0.001f);

                        totalLenght += safe_step;
                        pt_result = point_s + (totalLenght - start_t) * rayDirection;
                    }

                    if (pointFound) {
                        pt_result += (sdf_v * step_scale) * rayDirection;

                        sdf_v = readFromSDFInterpolated(pt_result);
                        pt_result += (sdf_v * step_scale) * rayDirection;

                        points_ptr[x] = pt_result * voxel_size_;
                    } else {
                        points_ptr[x](0) = std::numeric_limits<float>::quiet_NaN();
                    }
                }
            }
        });
}

void ReconstructionPipeline::allocateMemoryForVoxels(
    const Points& points, const Eigen::Matrix4f& camera_pose) {
    const Eigen::Matrix3f r = camera_pose.block(0, 0, 3, 3);
    const Eigen::Vector3f t = camera_pose.block(0, 3, 3, 1);

    const float oneOverVoxelSize = 1.0f / (voxel_size_ * VOXEL_BLOCK_SIZE);
    Points downsample_points;
    voxelDownSample(points, downsample_points);

    for (auto& point : downsample_points) {
        int nstep = 0;
        float norm = 0.0f;

        Eigen::Vector3f point_in_camera = point, direction;
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

            if (isValid(blockPos, depth_limit_)) {
                uint64_t code = encode(blockPos);
                if (allcator_.allocate(code)) {
                    updated_list_.emplace_back(code);
                }
            }
            point_s += direction;
        }
    }
}

void ReconstructionPipeline::integrate(const Points& points, const Eigen::Matrix4f& camera_pose) {
    const Eigen::Matrix3f r = camera_pose.block(0, 0, 3, 3).transpose();
    const Eigen::Vector3f t = camera_pose.block(0, 3, 3, 1);

    Timer timer("integrator");
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, updated_list_.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t id = range.begin(); id != range.end(); ++id) {
                Eigen::Vector3i blockPos = decode(updated_list_[id]);
                Voxel* current_voxel_block = allcator_.accessVoxelBlock(updated_list_[id]);
                for (size_t i = 0; i < coord_offsets_.size(); ++i) {
                    const Eigen::Vector3i& offset = coord_offsets_[i];
                    int localId =
                        offset.x() + offset.y() * EXPANDED_VOXEL_BLOCK_SIZE +
                        offset.z() * EXPANDED_VOXEL_BLOCK_SIZE * EXPANDED_VOXEL_BLOCK_SIZE;
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
                    if (std::abs(eta) > mu_) continue;

                    // 更新sdf值
                    float old_sdf = shortToFloat(current_voxel.sdf);
                    float old_weight = ushortToFloat(current_voxel.depth_weight);
                    if (old_weight == max_weight_) continue;
                    float new_sdf = std::min(1.0f, eta / mu_);
                    float new_weight = 1.0f;

                    new_sdf = old_weight * old_sdf + new_weight * new_sdf;
                    new_weight = new_weight + old_weight;
                    new_sdf /= new_weight;

                    new_weight = std::min(new_weight, max_weight_);

                    current_voxel.sdf = floatToShort(new_sdf);
                    current_voxel.depth_weight = floatToUshort(new_weight);
                }
            }
        });
}

void ReconstructionPipeline::voxelDownSample(
    const Points& origin_points, Points& processed_points) {
    processed_points.clear();
    processed_points.reserve(origin_points.size() >> 4);
    std::unordered_set<Eigen::Vector3i, Vector3iHash> voxel_map;
    voxel_map.reserve(static_cast<size_t>(std::pow(viewFrustum_max_ / voxel_size_, 2) / 2.0f));
    const float oneOverVoxelSize = 1.0f / (voxel_size_ * VOXEL_BLOCK_SIZE);
    for (const auto& p : origin_points) {
        if (std::isnan(p(0))) continue;
        const Eigen::Vector3i coord = (p.array() * oneOverVoxelSize).floor().cast<int>();
        if (voxel_map.insert(coord).second) {
            processed_points.push_back(p);
        }
    }
}

inline Eigen::Vector3i IntToFloor(const Eigen::Vector3f& in, Eigen::Vector3f& other) {
    Eigen::Vector3i tmp = in.array().floor().cast<int>();
    other = in - tmp.cast<float>();
    return tmp;
}

inline Eigen::Vector3i posToBlockPos(Eigen::Vector3i& point) {
    return Eigen::Vector3i(point.x() >> 3, point.y() >> 3, point.z() >> 3);
}

float ReconstructionPipeline::readFromSDFInterpolated(const Eigen::Vector3f& point) {
    Eigen::Vector3f coeff;
    Eigen::Vector3i position = IntToFloor(point, coeff);

    float v1, v2, res1, res2;
    const float cx = coeff(0);
    const float cy = coeff(1);
    const float cz = coeff(2);

    Eigen::Vector3i blockPos = posToBlockPos(position);
    if (!isValid(blockPos, depth_limit_)) return 1.0f;
    uint64_t code = encode(blockPos);
    const Voxel* current_voxel_block = allcator_.accessVoxelBlock(code);
    if (current_voxel_block == nullptr) {
        return 1.0f;
    }

    Eigen::Vector3i offset = position - blockPos * VOXEL_BLOCK_SIZE;
    v1 = readSDFByVoxelBlock(current_voxel_block, offset);
    v2 = readSDFByVoxelBlock(current_voxel_block, offset + Eigen::Vector3i(1, 0, 0));
    res1 = (1.0f - cx) * v1 + cx * v2;

    v1 = readSDFByVoxelBlock(current_voxel_block, offset + Eigen::Vector3i(0, 1, 0));
    v2 = readSDFByVoxelBlock(current_voxel_block, offset + Eigen::Vector3i(1, 1, 0));
    res1 = (1.0f - cy) * res1 + cy * ((1.0f - cx) * v1 + cx * v2);

    v1 = readSDFByVoxelBlock(current_voxel_block, offset + Eigen::Vector3i(0, 0, 1));
    v2 = readSDFByVoxelBlock(current_voxel_block, offset + Eigen::Vector3i(1, 0, 1));
    res2 = (1.0f - cx) * v1 + cx * v2;

    v1 = readSDFByVoxelBlock(current_voxel_block, offset + Eigen::Vector3i(0, 1, 1));
    v2 = readSDFByVoxelBlock(current_voxel_block, offset + Eigen::Vector3i(1, 1, 1));
    res2 = (1.0f - cy) * res2 + cy * ((1.0f - cx) * v1 + cx * v2);

    return (1.0f - cz) * res1 + cz * res2;
}

const Voxel ReconstructionPipeline::readVoxel(Eigen::Vector3i point) {
    Eigen::Vector3i blockPos = posToBlockPos(point);
    if (!isValid(blockPos, depth_limit_))
        return Voxel(
            std::numeric_limits<int16_t>::max(), std::numeric_limits<uint16_t>::min(), 0, 0, 0, 0);
    uint64_t code = encode(blockPos);
    const Voxel* current_voxel_block = allcator_.accessVoxelBlock(code);
    if (current_voxel_block == nullptr) {
        return Voxel(
            std::numeric_limits<int16_t>::max(), std::numeric_limits<uint16_t>::min(), 0, 0, 0, 0);
    }

    Eigen::Vector3i offset = point - blockPos * VOXEL_BLOCK_SIZE;
    return readVoxelByVoxelBlock(current_voxel_block, offset);
}
}  // namespace ScanReconstruction
