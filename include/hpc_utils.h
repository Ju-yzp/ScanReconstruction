#ifndef SCAN_RECONSTRUCTION_HPC_UTILS_H_
#define SCAN_RECONSTRUCTION_HPC_UTILS_H_

#include <Constants.h>
#include <Types.h>

// ipp
#include <ipp.h>

// cpp
#include <limits>

namespace ScanReconstruction {
// specfity design for voxel integrate
inline void vecShortToFloat(const short* source, float* target) {
    using Lim = std::numeric_limits<short>;
    constexpr static float scale = 1.0f / ((static_cast<float>(Lim::max())) + 1.0f);
    ippsConvert_16s32f(source, target, VOXEL_BLOCK_SIZE3);
    ippsMulC_32f_I(scale, target, VOXEL_BLOCK_SIZE3);
}

inline void vecUshortToFloat(const unsigned short* source, float* target) {
    constexpr static float scale = MAX_DEPTH_WEIGHT / (float)std::numeric_limits<uint16_t>::max();
    ippsConvert_16u32f(source, target, VOXEL_BLOCK_SIZE3);
    ippsMulC_32f_I(scale, target, VOXEL_BLOCK_SIZE3);
}

inline void vecFloatToShort(const float* source, short* target) {
    constexpr static int scaleFactor = -15;
    ippsConvert_32f16s_Sfs(source, target, VOXEL_BLOCK_SIZE3, ippRndFinancial, scaleFactor);
}

inline void vecFloatToUshort(const float* source, unsigned short* target) {
    alignas(32) float scaled[VOXEL_BLOCK_SIZE3];
    constexpr static float factor = std::numeric_limits<unsigned short>::max() / MAX_DEPTH_WEIGHT;
    ippsMulC_32f(source, factor, scaled, VOXEL_BLOCK_SIZE3);
    ippsConvert_32f16u_Sfs(scaled, (Ipp16u*)target, VOXEL_BLOCK_SIZE3, ippRndNear, 0);
}

inline void atomShortToFloat(short* source, float* target) {
    using Lim = std::numeric_limits<short>;
    constexpr static float scale = 1.0f / (static_cast<float>(Lim::max()));
    *target = static_cast<float>(*source) * scale;
}

inline void atomUshortToFloat(unsigned short* source, float* target) {
    constexpr float scale = MAX_DEPTH_WEIGHT / (float)std::numeric_limits<unsigned short>::max();
    *target = static_cast<float>(*source) * scale;
}

inline void atomFloatToShort(float* source, short* target) {
    using Lim = std::numeric_limits<short>;
    *target = static_cast<short>(std::clamp(
        (*source) * static_cast<float>(Lim::max()), static_cast<float>(Lim::min()),
        static_cast<float>(Lim::max())));
}

inline void atomFloatToUshort(float* source, unsigned short* target) {
    using Lim = std::numeric_limits<unsigned short>;
    const float scale = static_cast<float>(Lim::max()) / MAX_DEPTH_WEIGHT;
    float clamped = std::clamp(*source, 0.0f, MAX_DEPTH_WEIGHT);
    *target = static_cast<unsigned short>(std::round(clamped * scale));
}

inline void initVoxelBlock(VoxelBlock* voxel_block) {
    if (!voxel_block) return;
    short* s_ptr =
        reinterpret_cast<short*>(reinterpret_cast<char*>(voxel_block) + offsetof(VoxelBlock, sdf));
    unsigned short* w_ptr = reinterpret_cast<unsigned short*>(
        reinterpret_cast<char*>(voxel_block) + offsetof(VoxelBlock, weight));
    for (int i = 0; i < (int)VOXEL_BLOCK_SIZE3; ++i) {
        *(s_ptr + i) = std::numeric_limits<short>::max();
        *(w_ptr + i) = 1;
    }
}

inline void integrateWithVoxelBlock(
    VoxelBlock* voxel_block, const std::vector<float>& distances, const std::vector<float>& depth,
    const float mu) {
    if (!voxel_block || (depth.size() != distances.size()) || depth.size() != VOXEL_BLOCK_SIZE3)
        return;

    alignas(64) float old_sdf[VOXEL_BLOCK_SIZE3], new_sdf[VOXEL_BLOCK_SIZE3],
        old_weight[VOXEL_BLOCK_SIZE3], new_weight[VOXEL_BLOCK_SIZE3], abs_eta[VOXEL_BLOCK_SIZE3];

    short* sdf_ptr = reinterpret_cast<short*>(voxel_block->sdf);
    unsigned short* weight_ptr = reinterpret_cast<unsigned short*>(voxel_block->weight);
    const float* depth_ptr = depth.data();
    const float* dist_ptr = distances.data();
    const float inv_mu = 1.0f / mu;

    vecShortToFloat(sdf_ptr, old_sdf);
    vecUshortToFloat(weight_ptr, old_weight);

    ippsSub_32f(dist_ptr, depth_ptr, new_sdf, VOXEL_BLOCK_SIZE3);
    ippsAbs_32f(new_sdf, abs_eta, VOXEL_BLOCK_SIZE3);

    __m256 v_mu = _mm256_set1_ps(mu);
    __m256 v_max_weight = _mm256_set1_ps(MAX_DEPTH_WEIGHT);
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_zero = _mm256_setzero_ps();
    for (int i = 0; i < (int)VOXEL_BLOCK_SIZE3; i += 8) {
        __m256 v_eta = _mm256_loadu_ps(&abs_eta[i]);
        __m256 v_old = _mm256_loadu_ps(&old_weight[i]);
        __m256 mask1 = _mm256_cmp_ps(v_eta, v_mu, _CMP_LT_OS);
        __m256 mask2 = _mm256_cmp_ps(v_old, v_max_weight, _CMP_LT_OS);
        __m256 final_mask = _mm256_and_ps(mask1, mask2);

        __m256 res = _mm256_blendv_ps(v_zero, v_one, final_mask);
        _mm256_storeu_ps(&new_weight[i], res);
    }

    ippsMulC_32f(new_sdf, inv_mu, new_sdf, VOXEL_BLOCK_SIZE3);
    ippsMul_32f_I(new_weight, new_sdf, VOXEL_BLOCK_SIZE3);
    ippsMul_32f_I(old_weight, old_sdf, VOXEL_BLOCK_SIZE3);
    ippsAdd_32f_I(old_sdf, new_sdf, VOXEL_BLOCK_SIZE3);
    ippsAdd_32f_I(old_weight, new_weight, VOXEL_BLOCK_SIZE3);
    ippsDiv_32f_I(new_weight, new_sdf, VOXEL_BLOCK_SIZE3);

    vecFloatToShort(new_sdf, sdf_ptr);
    vecFloatToUshort(new_weight, weight_ptr);
}
}  // namespace ScanReconstruction
#endif
