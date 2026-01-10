#include <Scene.h>
#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace ScanReconstruction {

inline Eigen::Vector3i IntToFloor(const Eigen::Vector3f& in, Eigen::Vector3f& other) {
    Eigen::Vector3i tmp = in.array().floor().cast<int>();
    other = in - tmp.cast<float>();
    return tmp;
}

Scene::Scene(std::shared_ptr<GlobalSettings> global_settings)
    : voxel_hash_((size_t)global_settings->reverse_entries_num),
      voxel_size_(global_settings->voxel_size),
      entries_num_(global_settings->reverse_entries_num),
      free_entries_num_(global_settings->reverse_entries_num) {}

std::optional<HashEntry> Scene::insert(Eigen::Vector3i& blockPos) {
    int offset = VoxelHash::getHashIndex(blockPos);
    if (offset > entries_num_ - 1) return std::nullopt;

    HashEntry* entries_list = voxel_hash_.entries_ptr;
    Voxel* voxel_list = voxel_hash_.voxel_ptr;
    HashEntry& current_entry = entries_list[offset];
    if (current_entry.ptr == -1) {
        current_entry.ptr = entries_num_ - free_entries_num_;
        current_entry.pos = blockPos;
        --free_entries_num_;
        VoxelHash::initVoxelBlock(voxel_list + current_entry.ptr * VOLUME);
        return current_entry;
    }

    if (current_entry.pos == blockPos) return current_entry;

    HashEntry* hashEntry = &current_entry;
    while (hashEntry->offset != -1) {
        if (hashEntry->pos == blockPos) return *hashEntry;
        hashEntry = entries_list + hashEntry->offset;
    }

    if (hashEntry->pos == blockPos) return *hashEntry;
    if (free_entries_num_ == 0) return std::nullopt;
    int memory_offset = entries_num_ - free_entries_num_;
    --free_entries_num_;
    hashEntry->offset = memory_offset;
    entries_list[memory_offset].ptr = memory_offset;
    entries_list[memory_offset].pos = blockPos;
    VoxelHash::initVoxelBlock(voxel_list + memory_offset * VOLUME);
    return entries_list[memory_offset];
}

// constexpr int64_t BIT_14 = 14;
// constexpr int64_t MAX_DIFF_14 = 1LL << BIT_14;  // 16384

// inline int64_t cooedDiffFloatToUint64(float diff) {
//     float scaled = diff * static_cast<float>(MAX_DIFF_14);
//     return static_cast<int64_t>(std::clamp(scaled, 0.0f, static_cast<float>(MAX_DIFF_14)));
// }

// float Scene::readFromSDFInterpolated(const Eigen::Vector3f& point) {
//     Eigen::Vector3f coeff;
//     Eigen::Vector3i position = IntToFloor(point, coeff);

//     const int64_t cx = cooedDiffFloatToUint64(coeff(0));
//     const int64_t cy = cooedDiffFloatToUint64(coeff(1));
//     const int64_t cz = cooedDiffFloatToUint64(coeff(2));
//     const int64_t max_diff = MAX_DIFF_14;

//     int64_t v1, v2, r_y0, r_y1, res1, res2;

//     v1 = readVoxel(position + Eigen::Vector3i(0, 0, 0)).get_sdfWithInt64();
//     v2 = readVoxel(position + Eigen::Vector3i(1, 0, 0)).get_sdfWithInt64();
//     r_y0 = (max_diff - cx) * v1 + cx * v2;

//     v1 = readVoxel(position + Eigen::Vector3i(0, 1, 0)).get_sdfWithInt64();
//     v2 = readVoxel(position + Eigen::Vector3i(1, 1, 0)).get_sdfWithInt64();
//     r_y1 = (max_diff - cx) * v1 + cx * v2;

//     res1 = (max_diff - cy) * r_y0 + cy * r_y1;

//     v1 = readVoxel(position + Eigen::Vector3i(0, 0, 1)).get_sdfWithInt64();
//     v2 = readVoxel(position + Eigen::Vector3i(1, 0, 1)).get_sdfWithInt64();
//     r_y0 = (max_diff - cx) * v1 + cx * v2;

//     v1 = readVoxel(position + Eigen::Vector3i(0, 1, 1)).get_sdfWithInt64();
//     v2 = readVoxel(position + Eigen::Vector3i(1, 1, 1)).get_sdfWithInt64();
//     r_y1 = (max_diff - cx) * v1 + cx * v2;

//     res2 = (max_diff - cy) * r_y0 + cy * r_y1;

//     int64_t final_sdf_64 = (max_diff - cz) * res1 + cz * res2;

//     return static_cast<float>(final_sdf_64 >> 42) / 32768.0f;
// }

// float Scene::readWithConfidenceFromSDFInterpolated(const Eigen::Vector3f& point) {
//     Voxel voxel;
//     int64_t res1, res2, v1, v2;

//     Eigen::Vector3f coeff;
//     Eigen::Vector3i position = IntToFloor(point, coeff);

//     const int64_t cx = cooedDiffFloatToUint64(coeff(0));
//     const int64_t cy = cooedDiffFloatToUint64(coeff(1));
//     const int64_t cz = cooedDiffFloatToUint64(coeff(2));

//     const int64_t max_diff = MAX_DIFF_14;

//     voxel = readVoxel(position + Eigen::Vector3i(0, 0, 0));
//     v1 = voxel.get_sdfWithInt64();
//     voxel = readVoxel(position + Eigen::Vector3i(1, 0, 0));
//     v2 = voxel.get_sdfWithInt64();
//     int64_t r1_y0 = (max_diff - cx) * v1 + cx * v2;

//     voxel = readVoxel(position + Eigen::Vector3i(0, 1, 0));
//     v1 = voxel.get_sdfWithInt64();
//     voxel = readVoxel(position + Eigen::Vector3i(1, 1, 0));
//     v2 = voxel.get_sdfWithInt64();
//     int64_t r1_y1 = (max_diff - cx) * v1 + cx * v2;

//     res1 = (max_diff - cy) * r1_y0 + cy * r1_y1;

//     voxel = readVoxel(position + Eigen::Vector3i(0, 0, 1));
//     v1 = voxel.get_sdfWithInt64();
//     voxel = readVoxel(position + Eigen::Vector3i(1, 0, 1));
//     v2 = voxel.get_sdfWithInt64();
//     int64_t r2_y0 = (max_diff - cx) * v1 + cx * v2;

//     voxel = readVoxel(position + Eigen::Vector3i(0, 1, 1));
//     v1 = voxel.get_sdfWithInt64();
//     voxel = readVoxel(position + Eigen::Vector3i(1, 1, 1));
//     v2 = voxel.get_sdfWithInt64();
//     int64_t r2_y1 = (max_diff - cx) * v1 + cx * v2;

//     res2 = (max_diff - cy) * r2_y0 + cy * r2_y1;

//     int64_t final_sdf_64 = (max_diff - cz) * res1 + cz * res2;

//     float normalized_sdf = static_cast<float>(final_sdf_64 >> 42) / 32768.0f;

//     return normalized_sdf;
// }

inline float readSDFByVoxelBlock(const Voxel* voxel_block, Eigen::Vector3i position) {
    position += Eigen::Vector3i(1, 1, 1);
    return voxel_block
        [position.x() + position.y() * EXPAND_BOUND + position.z() * EXPAND_BOUND * EXPAND_BOUND]
            .get_sdf();
}

inline Voxel readVoxelByVoxelBlock(const Voxel* voxel_block, Eigen::Vector3i position) {
    position += Eigen::Vector3i(1, 1, 1);
    return voxel_block
        [position.x() + position.y() * EXPAND_BOUND + position.z() * EXPAND_BOUND * EXPAND_BOUND];
}

float Scene::readFromSDFInterpolated(const Eigen::Vector3f& point) {
    Eigen::Vector3f coeff;
    Eigen::Vector3i position = IntToFloor(point, coeff);

    float v1, v2, res1, res2;
    const float cx = coeff(0);
    const float cy = coeff(1);
    const float cz = coeff(2);

    Eigen::Vector3i blockPos = voxel_hash_.posToBlockPos(position);
    const Voxel* curret_voxel_block = readVoxelBlock(blockPos);
    if (curret_voxel_block == nullptr) {
        return 1.0f;
    }

    Eigen::Vector3i offset = position - blockPos * VOXEL_BLOCK_SIZE;
    v1 = readSDFByVoxelBlock(curret_voxel_block, offset);
    v2 = readSDFByVoxelBlock(curret_voxel_block, offset + Eigen::Vector3i(1, 0, 0));
    res1 = (1.0f - cx) * v1 + cx * v2;

    v1 = readSDFByVoxelBlock(curret_voxel_block, offset + Eigen::Vector3i(0, 1, 0));
    v2 = readSDFByVoxelBlock(curret_voxel_block, offset + Eigen::Vector3i(1, 1, 0));
    res1 = (1.0f - cy) * res1 + cy * ((1.0f - cx) * v1 + cx * v2);

    v1 = readSDFByVoxelBlock(curret_voxel_block, offset + Eigen::Vector3i(0, 0, 1));
    v2 = readSDFByVoxelBlock(curret_voxel_block, offset + Eigen::Vector3i(1, 0, 1));
    res2 = (1.0f - cx) * v1 + cx * v2;

    v1 = readSDFByVoxelBlock(curret_voxel_block, offset + Eigen::Vector3i(0, 1, 1));
    v2 = readSDFByVoxelBlock(curret_voxel_block, offset + Eigen::Vector3i(1, 1, 1));
    res2 = (1.0f - cy) * res2 + cy * ((1.0f - cx) * v1 + cx * v2);

    return (1.0f - cz) * res1 + cz * res2;
}

float Scene::readWithConfidenceFromSDFInterpolated(const Eigen::Vector3f& point) {
    Eigen::Vector3f coeff;
    Eigen::Vector3i position = IntToFloor(point, coeff);

    float v1, v2, res1, res2;
    const float cx = coeff(0);
    const float cy = coeff(1);
    const float cz = coeff(2);

    Eigen::Vector3i blockPos = voxel_hash_.posToBlockPos(position);
    const Voxel* curret_voxel_block = readVoxelBlock(blockPos);
    if (curret_voxel_block == nullptr) {
        return std::numeric_limits<int16_t>::max();
    }

    Eigen::Vector3i offset = position - blockPos * VOXEL_BLOCK_SIZE;
    v1 = readSDFByVoxelBlock(curret_voxel_block, offset);
    v2 = readSDFByVoxelBlock(curret_voxel_block, offset + Eigen::Vector3i(1, 0, 0));
    res1 = (1.0f - cx) * v1 + cx * v2;

    v1 = readSDFByVoxelBlock(curret_voxel_block, offset + Eigen::Vector3i(0, 1, 0));
    v2 = readSDFByVoxelBlock(curret_voxel_block, offset + Eigen::Vector3i(1, 1, 0));
    res1 = (1.0f - cy) * res1 + cy * ((1.0f - cx) * v1 + cx * v2);

    v1 = readSDFByVoxelBlock(curret_voxel_block, offset + Eigen::Vector3i(0, 0, 1));
    v2 = readSDFByVoxelBlock(curret_voxel_block, offset + Eigen::Vector3i(1, 0, 1));
    res2 = (1.0f - cx) * v1 + cx * v2;

    v1 = readSDFByVoxelBlock(curret_voxel_block, offset + Eigen::Vector3i(0, 1, 1));
    v2 = readSDFByVoxelBlock(curret_voxel_block, offset + Eigen::Vector3i(1, 1, 1));
    res2 = (1.0f - cy) * res2 + cy * ((1.0f - cx) * v1 + cx * v2);

    return (1.0f - cz) * res1 + cz * res2;
}

const Voxel Scene::readVoxel(Eigen::Vector3i point) {
    Eigen::Vector3i blockPos = voxel_hash_.posToBlockPos(point);
    int hashId = VoxelHash::getHashIndex(blockPos);

    HashEntry* hash_entries_list = voxel_hash_.entries_ptr;

    HashEntry* entry_ptr = &hash_entries_list[hashId];
    Eigen::Vector3i& pos = entry_ptr->pos;
    while (pos != blockPos) {
        if (entry_ptr->offset == -1)
            return Voxel(std::numeric_limits<int16_t>::max(), std::numeric_limits<uint16_t>::min());
        entry_ptr = &hash_entries_list[entry_ptr->offset];
        pos = entry_ptr->pos;
    }

    if (entry_ptr->ptr == -1)
        return Voxel(std::numeric_limits<int16_t>::max(), std::numeric_limits<uint16_t>::min());
    else {
        Eigen::Vector3i offset = point - blockPos * VOXEL_BLOCK_SIZE;
        Voxel* current_voxel_block = voxel_hash_.voxel_ptr + entry_ptr->ptr * VOLUME;
        return readVoxelByVoxelBlock(current_voxel_block, offset);
    }
}

const Voxel* Scene::readVoxelBlock(Eigen::Vector3i blockPos) {
    int hashId = VoxelHash::getHashIndex(blockPos);

    HashEntry* hash_entries_list = voxel_hash_.entries_ptr;

    HashEntry* entry_ptr = &hash_entries_list[hashId];
    Eigen::Vector3i& pos = entry_ptr->pos;
    while (pos != blockPos) {
        if (entry_ptr->offset == -1) return nullptr;
        entry_ptr = &hash_entries_list[entry_ptr->offset];
        pos = entry_ptr->pos;
    }

    if (entry_ptr->ptr == -1)
        return nullptr;
    else
        return voxel_hash_.voxel_ptr + entry_ptr->ptr * VOLUME;
}
}  // namespace ScanReconstruction
