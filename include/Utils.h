#ifndef UTILS_H_
#define UTILS_H_

#include <tsl/robin_growth_policy.h>
#include <tsl/robin_map.h>
#include <Eigen/Core>
#include <cmath>

namespace ScanReconstruction {

// // 体素键
// using VoxelKey = Eigen::Vector3i;

// // 体素
// struct Voxel {
//     float weight{0.0f};
//     float sdf{1.0f};
// };

// // 体素块
// struct VoxelBlock {
//     std::atomic<bool> state;
//     std::shared_ptr<Voxel> data;
// };

// // 哈希函数
// struct VoxelKeyHash {
//     std::size_t operator()(const Eigen::Vector3i& v) const {
//         size_t seed = 0;
//         seed ^= std::hash<int>{}(v.x()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//         seed ^= std::hash<int>{}(v.y()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//         seed ^= std::hash<int>{}(v.z()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//         return seed;
//     }
// };

// // 哈希键值比对函数
// struct VoxelKeyEqual {
//     bool operator()(const Eigen::Vector3i& a, const Eigen::Vector3i& b) const {
//         return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
//     }
// };

// // 体素哈希表
// using VoxelMap = tsl::robin_map<
//     VoxelKey, VoxelBlock, VoxelKeyHash, VoxelKeyEqual,
//     std::allocator<std::pair<VoxelKey, VoxelBlock>>, false,
//     tsl::rh::power_of_two_growth_policy<2>>;

// 用于点云图或者法向量图
using Map = std::vector<Eigen::Vector3f>;

#ifndef MIN
#define MIN(a, b) ((a < b) ? a : b)
#endif

#ifndef MAX
#define MAX(a, b) ((a < b) ? b : a)
#endif

#ifndef ABS
#define ABS(a) ((a < 0) ? -a : a)
#endif

#ifndef CLAMP
#define CLAMP(x, a, b) MAX((a), MIN((b), (x)))
#endif

inline Eigen::Matrix3f skew(const Eigen::Vector3f v) {
    Eigen::Matrix3f m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
}

inline float rho(float r, float huber_b) {
    float tmp = std::fabs(r) - huber_b;
    tmp = std::max(tmp, 0.0f);
    return r * r - tmp * tmp;
}

inline float rho_deriv(float r, float huber_b) { return 2.0f * CLAMP(r, -huber_b, huber_b); }

inline float rho_deriv2(float r, float huber_b) { return fabs(r) < huber_b ? 2.0f : 0.0f; }

}  // namespace ScanReconstruction
#endif
