#ifndef SACN_RECONSTRUCTION_INTERPOLATE_UTILS_H_
#define SACN_RECONSTRUCTION_INTERPOLATE_UTILS_H_

#include <Constants.h>
#include <Types.h>
#include <scene.h>
#include <Eigen/Eigen>

namespace ScanReconstruction {
inline float readSdfByVoxelBlock(VoxelBlock* voxel_block, const Eigen::Vector3i& offset) {
    if (!voxel_block) return 1.0f;
    uint32_t code = _pdep_u32((uint32_t)offset(0), 0x49) | _pdep_u32((uint32_t)offset(1), 0x92) |
                    _pdep_u32((uint32_t)offset(2), 0x124);
    return voxel_block->sdf[code];
}

// 三線性插值
inline float readSdfWithInterpolated(Eigen::Vector3f position, const Scene* scene) {
    Eigen::Vector3i t_position = position.array().floor().cast<int>();
    Eigen::Vector3f coeff = position - t_position.cast<float>();
    const float cx = coeff(0);
    const float cy = coeff(1);
    const float cz = coeff(2);

    const static std::vector<Eigen::Vector3i> offsets = {
        Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(1, 0, 0), Eigen::Vector3i(0, 1, 0),
        Eigen::Vector3i(1, 1, 0), Eigen::Vector3i(0, 0, 1), Eigen::Vector3i(1, 0, 1),
        Eigen::Vector3i(0, 1, 1), Eigen::Vector3i(1, 1, 1)};

    Eigen::Vector3i new_position[8], offset_in_block[8], blockPos[8];
    alignas(64) float sdf[8];
    VoxelBlock *current_voxel_block = nullptr, *first_voxel_block = nullptr;
    first_voxel_block = scene->findVoxelBlock(blockPos[0]);
    sdf[0] = readSdfByVoxelBlock(first_voxel_block, offset_in_block[0]);

    for (int i = 1; i < 8; ++i) {
        new_position[i] = t_position + offsets[i];
        blockPos[i] = Eigen::Vector3i(
            new_position[i].x() >> 3, new_position[i].y() >> 3, new_position[i].z() >> 3);
        offset_in_block[i] = new_position[i] - blockPos[i] * VOXEL_BLOCK_SIZE;
        current_voxel_block =
            blockPos[i] == blockPos[0] ? first_voxel_block : scene->findVoxelBlock(blockPos[i]);
        sdf[i] = readSdfByVoxelBlock(current_voxel_block, offset_in_block[i]);
    }

    Eigen::Matrix2f m1, m2;
    Eigen::Vector2f dx(1.0f - cx, cx);
    m1 << sdf[0], sdf[1], sdf[4], sdf[5];
    m2 << sdf[2], sdf[3], sdf[6], sdf[7];
    return (Eigen::RowVector2f(1.0f - cz, cz) * ((1.0f - cy) * (m1 * dx) + cy * (m2 * dx))).value();
}
}  // namespace ScanReconstruction

#endif
