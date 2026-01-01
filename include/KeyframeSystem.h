#ifndef KEYFRAME_SYSTEM_H_
#define KEYFRAME_SYSTEM_H_

// cpp
#include <cstddef>
#include <cstdint>

// sophus
#include <sophus/se3.hpp>

namespace ScanReconstruction {

using Normals = std::vector<Eigen::Vector3f>;
using Points = std::vector<Eigen::Vector3f>;

struct Histogram {
    std::vector<std::vector<uint32_t>> ndHistogram;
    Histogram(size_t n) { ndHistogram.resize(n, std::vector<uint32_t>(n, 0)); }
};

struct Keyframe {
    Keyframe(size_t n) : histogram(n) {}

    // 法向量-深度联合直方图
    Histogram histogram;

    // 帧在子地图中的位姿
    Sophus::SE3f pose = Sophus::SE3f();

    // 地图索引
    int map_id = -1;
};

class KeyframeSystem {
public:
    explicit KeyframeSystem(
        int width, int height, size_t bin_num, float similarity_threshold, float entropy_threshold,
        float viewFrustum_max, float viewFrustum_min);

    void searchKNearsetNeighbor(std::vector<size_t>& frames, int n, Keyframe frame);

    void computeKeyFrame(Normals& normals, Points& points, Keyframe& keyframe);

    bool insert(Keyframe& new_keyframe);

    Keyframe get_keyframe(size_t id) { return keyframes_[id]; }

    size_t get_bin_num() { return bin_num_; }

    size_t size() { return keyframes_.size(); }

private:
    void updateHistogram(
        const Eigen::Vector3f& point, const Eigen::Vector3f& normal, Histogram& histogram);

    bool isSimilarity(Keyframe& a, Keyframe& b, float& distance);

    bool hasEnoughInformation(Histogram& histogram);

    void computeChiSquare(Histogram& a, Histogram& b, float& chi_square);

    int width_, height_;

    size_t bin_num_;

    std::vector<Keyframe> keyframes_;

    // 相似度閾值
    float similarty_threshold_;

    // 信息熵閾值
    float entropy_threshold_;

    float viewFrustum_max_, viewFrustum_min_;
    float depth_step_;

    std::vector<Eigen::Vector3f> precomputed_points;
};
}  // namespace ScanReconstruction

#endif
