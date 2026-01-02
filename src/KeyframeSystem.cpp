// scan_reconstruction
#include <KeyframeSystem.h>

// tbb
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/partitioner.h>

// cpp
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace ScanReconstruction {
KeyframeSystem::KeyframeSystem(
    int width, int height, size_t bin_num, float similarity_threshold, float entropy_threshold,
    float viewFrustum_max, float viewFrustum_min)
    : width_(width),
      height_(height),
      bin_num_(bin_num),
      similarty_threshold_(similarity_threshold),
      entropy_threshold_(entropy_threshold),
      viewFrustum_max_(viewFrustum_max),
      viewFrustum_min_(viewFrustum_min) {
    precomputed_points.resize(bin_num_, Eigen::Vector3f::Zero());
    float golden_ratio = (1.0f + std::sqrt(5.0f)) / 2.0f;
    float angle_increment = 2.0f * M_PIf * (1.0f - 1.0f / golden_ratio);

    for (int i = 0; i < (int)bin_num_; ++i) {
        float t = (float)i / (float)(bin_num_ - 1);
        float z = 1.0f - 2.0f * t;

        float radius = std::sqrt(1.0f - z * z);

        float theta = angle_increment * (float)i;

        float x = std::cos(theta) * radius;
        float y = std::sin(theta) * radius;

        precomputed_points[size_t(i)] = Eigen::Vector3f(x, y, z);
    }

    depth_step_ = (viewFrustum_max_ - viewFrustum_min_) / (float)(bin_num_ - 1);
}

void KeyframeSystem::computeKeyFrame(Normals& normals, Points& points, Keyframe& keyframe) {
    // if (keyframe.map_id == -1) throw std::runtime_error("Invalid map id");

    int total_num = width_ * height_;

    if (total_num != (int)normals.size() || total_num != (int)points.size())
        throw std::runtime_error("Input size mismatch with width and height");

    int start_row = int((float)height_ * 0.2f);
    int end_row = height_ - int((float)height_ * 0.2f);
    int start_col = int((float)width_ * 0.2f);
    int end_col = width_ - int((float)width_ * 0.2f);
    Histogram total_histogram = tbb::parallel_reduce(
        tbb::blocked_range2d<int>(start_row, end_row, start_col, end_col),
        Histogram(this->bin_num_),
        [&](const tbb::blocked_range2d<int>& r, Histogram histogram) {
            for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
                const Eigen::Vector3f* points_ptr = points.data() + (y * width_);
                const Eigen::Vector3f* normals_ptr = normals.data() + (y * width_);
                for (int x = r.cols().begin(); x < r.cols().end(); ++x)
                    updateHistogram(points_ptr[x], normals_ptr[x], histogram);
            }

            return histogram;
        },
        [&](Histogram a, Histogram b) {
            for (size_t n_id = 0; n_id < bin_num_; ++n_id)
                for (size_t d_id = 0; d_id < bin_num_; ++d_id)
                    a.ndHistogram[n_id][d_id] += b.ndHistogram[n_id][d_id];

            return a;
        });

    keyframe.histogram.ndHistogram.swap(total_histogram.ndHistogram);
}

void KeyframeSystem::updateHistogram(
    const Eigen::Vector3f& point, const Eigen::Vector3f& normal, Histogram& histogram) {
    float z = point(2);
    if (std::isnan(point(0)) || z >= viewFrustum_max_ || z <= viewFrustum_min_) return;

    size_t d_id = size_t((z - viewFrustum_min_) / (depth_step_));
    d_id = std::min(d_id, bin_num_ - 1);
    if (std::isnan(normal(0))) return;
    float max_similarity = -1.0f;
    size_t best_bin = 0;

    for (size_t i = 0; i < (size_t)bin_num_; ++i) {
        float dot = normal.dot(precomputed_points[i]);
        if (dot > max_similarity) {
            max_similarity = dot;
            best_bin = i;
        }
    }
    ++histogram.ndHistogram[best_bin][d_id];
}

bool KeyframeSystem::insert(Keyframe& new_keyframe) {
    float distance{0.0f};

    if (!hasEnoughInformation(new_keyframe.histogram)) return false;

    for (size_t id = 0; id < keyframes_.size(); ++id)
        if (isSimilarity(new_keyframe, keyframes_[id], distance)) return false;

    keyframes_.emplace_back(new_keyframe);
    return true;
}

// 使用卡方檢驗來判斷兩幀的相似度
bool KeyframeSystem::isSimilarity(Keyframe& a, Keyframe& b, float& distance) {
    Histogram& a_histogram = a.histogram;
    Histogram& b_histogram = b.histogram;

    float chi_square = 0.0f;
    computeChiSquare(a_histogram, b_histogram, chi_square);

    distance = chi_square;
    return chi_square < similarty_threshold_;
}

void KeyframeSystem::searchKNearsetNeighbor(std::vector<size_t>& frames, int n, Keyframe frame) {
    struct FrameDistance {
        size_t id;
        float distance;
    };

    std::vector<FrameDistance> distances;
    for (size_t id = 0; id < keyframes_.size(); ++id) {
        FrameDistance fd;
        if (isSimilarity(keyframes_[id], frame, fd.distance)) {
            fd.id = id;
            distances.emplace_back(fd);
        }
    }

    std::sort(
        distances.begin(), distances.end(),
        [](const FrameDistance& a, const FrameDistance& b) { return a.distance < b.distance; });

    frames.clear();
    n = std::min(n, int(distances.size()));
    for (size_t id = 0; id < (size_t)n; ++id) frames.emplace_back(distances[id].id);
}

bool KeyframeSystem::hasEnoughInformation(Histogram& histogram) {
    float total_entropy{0.0f}, total_num{0.0f};
    for (size_t n_id = 0; n_id < bin_num_; ++n_id)
        for (size_t d_id = 0; d_id < bin_num_; ++d_id)
            total_num += (float)histogram.ndHistogram[n_id][d_id];

    if (total_num > 0)
        for (size_t n_id = 0; n_id < bin_num_; ++n_id) {
            std::vector<uint32_t> depth_histogram(bin_num_, 0);
            depth_histogram = histogram.ndHistogram[n_id];
            for (size_t d_id = 0; d_id < (size_t)bin_num_; ++d_id) {
                float num = (float)depth_histogram[d_id];
                if (num == 0) continue;
                float pi = (float)num / (float)total_num;
                total_entropy += -pi * std::log2(pi);
            }
        }
    std::cout << "Total points: " << total_num << std::endl;
    std::cout << "Entropy: " << total_entropy << std::endl;
    return total_entropy > entropy_threshold_;
}

void KeyframeSystem::computeChiSquare(Histogram& a, Histogram& b, float& chi_square) {
    chi_square = 0.0f;
    float sum_a = 0.0f, sum_b = 0.0f;

    for (size_t n_id = 0; n_id < bin_num_; ++n_id) {
        for (size_t d_id = 0; d_id < bin_num_; ++d_id) {
            sum_a += (float)a.ndHistogram[n_id][d_id];
            sum_b += (float)b.ndHistogram[n_id][d_id];
        }
    }

    for (size_t n_id = 0; n_id < bin_num_; ++n_id) {
        for (size_t d_id = 0; d_id < bin_num_; ++d_id) {
            float p_a = sum_a > 0.0f ? (float)a.ndHistogram[n_id][d_id] / sum_a : 0.0f;
            float p_b = sum_b > 0.0f ? (float)b.ndHistogram[n_id][d_id] / sum_b : 0.0f;

            if (p_a + p_b > 1e-6f) {
                chi_square += ((p_a - p_b) * (p_a - p_b)) / (p_a + p_b);
            }
        }
    }
}
}  // namespace ScanReconstruction
