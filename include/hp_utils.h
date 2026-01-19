#ifndef SCAN_RECONSTRUCTION_HP_UTILS_H_
#define SCAN_RECONSTRUCTION_HP_UTILS_H_

#include <Constants.h>
#include <Types.h>

#include <immintrin.h>

// cpp
#include <limits>

namespace ScanReconstruction {
inline void atomShortToFloat(short* source, float* target) {
    using Lim = std::numeric_limits<short>;
    constexpr static float scale = 1.0f / (static_cast<float>(Lim::max() + 1.0f));
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
    VoxelBlock* voxel_block, std::vector<float>& distances, std::vector<float>& depth,
    const float mu) {
    if (!voxel_block || (depth.size() != distances.size()) || depth.size() != VOXEL_BLOCK_SIZE3)
        return;
    short* sdf_ptr = voxel_block->sdf;
    unsigned short* weight_ptr = voxel_block->weight;
    float* depth_ptr = depth.data();
    float* dist_ptr = distances.data();

    // 縮放係數
    static const __m256 sdf_sf_scale =
        _mm256_set1_ps(1.0f / static_cast<float>(std::numeric_limits<short>::max()));
    static const __m256 sdf_fs_scale =
        _mm256_set1_ps(static_cast<float>(std::numeric_limits<short>::max()));
    static const __m256 weight_usf_scale = _mm256_set1_ps(
        MAX_DEPTH_WEIGHT / static_cast<float>(std::numeric_limits<unsigned short>::max()));
    static const __m256 weight_fus_scale = _mm256_set1_ps(
        static_cast<float>(std::numeric_limits<unsigned short>::max()) / MAX_DEPTH_WEIGHT);

    __m256 old_sdf, new_sdf, old_weight, new_weight, abs_eta;
    static const __m256 v_abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 v_mu = _mm256_set1_ps(mu);
    __m256 inv_mu = _mm256_set1_ps(1.0f / mu);
    static const __m256 v_max_weight = _mm256_set1_ps(MAX_DEPTH_WEIGHT);
    static const __m256 v_one = _mm256_set1_ps(1.0f);
    static const __m256 v_zero = _mm256_setzero_ps();

    for (int i = 0; i < (int)VOXEL_BLOCK_SIZE3; i += 8) {
        // short轉float
        __m128i sdf_short = _mm_loadu_si128((__m128i*)(sdf_ptr + i));
        __m256i v_int32 = _mm256_cvtepi16_epi32(sdf_short);
        old_sdf = _mm256_cvtepi32_ps(v_int32);
        old_sdf = _mm256_mul_ps(old_sdf, sdf_sf_scale);

        // ushort轉float
        __m128i v_ushort = _mm_loadu_si128((__m128i*)(weight_ptr + i));
        __m256i v_uint32 = _mm256_cvtepu16_epi32(v_ushort);
        old_weight = _mm256_cvtepi32_ps(v_uint32);
        old_weight = _mm256_mul_ps(old_weight, weight_usf_scale);

        // 計算eta
        __m256 v_depth = _mm256_loadu_ps(depth_ptr + i);
        __m256 v_dist = _mm256_loadu_ps(dist_ptr + i);
        new_sdf = _mm256_sub_ps(v_depth, v_dist);
        abs_eta = _mm256_and_ps(new_sdf, v_abs_mask);

        // 計算新權重
        __m256 mask1 = _mm256_cmp_ps(abs_eta, v_mu, _CMP_LT_OS);
        __m256 mask2 = _mm256_cmp_ps(old_weight, v_max_weight, _CMP_LT_OS);
        __m256 final_mask = _mm256_and_ps(mask1, mask2);
        new_weight = _mm256_blendv_ps(v_zero, v_one, final_mask);

        // 更新權重和sdf
        new_sdf = _mm256_mul_ps(new_sdf, inv_mu);
        old_sdf = _mm256_mul_ps(old_sdf, old_weight);
        new_sdf = _mm256_fmadd_ps(new_sdf, new_weight, old_sdf);
        new_weight = _mm256_add_ps(old_weight, new_weight);
        new_sdf = _mm256_div_ps(new_sdf, new_weight);

        new_sdf = _mm256_mul_ps(new_sdf, sdf_fs_scale);
        new_weight = _mm256_mul_ps(new_weight, weight_fus_scale);

        v_int32 = _mm256_cvtps_epi32(new_sdf);
        v_uint32 = _mm256_cvtps_epi32(new_weight);

        // 轉換后寫回緩存
        __m256i v_sdf_packed = _mm256_packs_epi32(v_int32, v_int32);
        __m256i v_w_packed = _mm256_packus_epi32(v_uint32, v_uint32);
        v_sdf_packed = _mm256_permute4x64_epi64(v_sdf_packed, _MM_SHUFFLE(3, 1, 2, 0));
        v_w_packed = _mm256_permute4x64_epi64(v_w_packed, _MM_SHUFFLE(3, 1, 2, 0));

        _mm_storeu_si128((__m128i*)(sdf_ptr + i), _mm256_castsi256_si128(v_sdf_packed));
        _mm_storeu_si128((__m128i*)(weight_ptr + i), _mm256_castsi256_si128(v_w_packed));
    }
}
}  // namespace ScanReconstruction
#endif
