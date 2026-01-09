#ifndef GLOBAL_SETTINGS_H_
#define GLOBAL_SETTINGS_H_

#include <tbb/global_control.h>

#include <Eigen/Dense>

namespace ScanReconstruction {
struct GlobalSettings {
    void set_max_num_threads(int max_num_threads) {
        static const auto tbb_control_settings = tbb::global_control(
            tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_num_threads));
    }

    // 地圖參數:
    //
    float visiable_threshold;

    // 最大權重
    float max_weight;

    // 截斷距離
    float mu;

    // 相机参数
    // 视锥体范围
    float viewFrustum_max, viewFrustum_min;

    int width, height;

    Eigen::Matrix3f k;

    // 地圖參數:
    // 体素分辨率
    float voxel_size;

    // 提前申请的体素块数量
    int reverse_entries_num;

    // 关键帧系统参数:
    // 球面划分bin
    size_t bin_num;

    // 相似度阈值
    float similarity_threshold;

    // 追踪器参数:
    // 最大迭代数
    int max_num_iterations, min_num_iterations;

    // 金字塔层数
    size_t pyramid_levels;

    // 空间距离最小，最大值，用于icp的鲁棒核，剔除异常点
    float space_threshold_min, space_threshold_max;

    // LM算法中的初始lambda值和缩放因子
    float lamdba_scale;
    float initial_lamdba;
};
}  // namespace ScanReconstruction

#endif
