#ifndef GLOBAL_SETTINGS_H_
#define GLOBAL_SETTINGS_H_

#include <tbb/global_control.h>

namespace ScanReconstrcution {
struct GlobalSettings {
    GlobalSettings(int max_num_threads) {
        static const auto tbb_control_settings = tbb::global_control(
            tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_num_threads));
    }

    // 相机参数:
    // 视锥体范围
    float viewFrustum_max, viewFrustum_min;

    int width;

    int height;

    // 地圖參數:
    // 体素分辨率
    float voxel_size;

    // 提前申请的体素块数量
    int reverse_num;

    // 体素块大小
    int voxel_block_size;

    // 关键帧系统参数:
    // 球面划分bin

    // 相似度阈值
    float similarity_threshold;

    // 追蹤器參數:
    // 最大迭代數
    int max_num_iterations;

    // 金字塔層數
    int pyramid_levels;
};
}  // namespace ScanReconstrcution

#endif
